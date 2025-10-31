import argparse
import json
import os
import re
import time
from typing import List, Optional, Tuple

import requests
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig

import wikienv, wrappers
from utils.prompter import Prompter
from uncertainty_utils import cal_uncertainty_entropy, cal_uncertainty_mean, cal_uncertainty_log_sum, cal_uncertainty_min, cal_uncertainty_norm


def build_model_and_tokenizer(base_model: str, load_in_4bit: bool = False):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    model.eval()
    return model, tokenizer


def llama2_prompt(model, tokenizer, instruction: str, input_text: str, temperature=0, top_p=1, num_beams=1, do_sample=False, max_new_tokens=64, return_probs=False):
    prompter = Prompter("llama2")
    prompt = prompter.generate_prompt(instruction, input_text)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        do_sample=do_sample,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    input_length = inputs.input_ids.shape[1]
    generated_tokens = generation_output.sequences[:, input_length:]
    output_text = tokenizer.decode(generated_tokens[0])

    if return_probs:
        transition_scores = model.compute_transition_scores(
            generation_output.sequences, generation_output.scores, normalize_logits=True
        )
        prob_dicts = []
        for tok, score in zip(generated_tokens[0], transition_scores[0]):
            prob_dicts.append({tokenizer.decode(tok): score.cpu().tolist()})
        return output_text, prob_dicts
    return output_text


def parse_actions(actions_str: str) -> List[str]:
    # Split by semicolon or newline
    parts = re.split(r";|\n", actions_str)
    return [p.strip() for p in parts if p.strip()]


def safe_env_step(env, action: str) -> Tuple[str, int, bool, dict]:
    attempts = 0
    start_ts = time.time()
    while attempts < 10:
        try:
            obs, r, done, info = env.step(action)
            latency_ms = int((time.time() - start_ts) * 1000)
            if isinstance(info, dict):
                info["latency_ms"] = latency_ms
            return obs, r, done, info
        except requests.exceptions.Timeout:
            attempts += 1
    return "", 0, False, {}


def collect_span_token_scores(generated_text: str, token_prob_seq: List[dict], start_char='[', end_char=']') -> Tuple[List[float], str]:
    # Build ordered list of (token_str, score)
    token_seq: List[Tuple[str, float]] = []
    for d in token_prob_seq:
        for k, v in d.items():
            token_seq.append((k, float(v)))

    start_idx = generated_text.find(start_char)
    end_idx = generated_text.find(end_char, start_idx + 1) if start_idx != -1 else -1

    if start_idx != -1 and end_idx != -1:
        answer_text = generated_text[start_idx + 1:end_idx]
        capture_start = start_idx + 1
        capture_end = end_idx
    else:
        nl = generated_text.find('\n')
        if nl == -1:
            answer_text = generated_text.strip()
            capture_start = 0
            capture_end = len(generated_text)
        else:
            answer_text = generated_text[:nl].strip()
            capture_start = 0
            capture_end = nl

    answer_token_probs: List[float] = []
    built = ""
    pos = 0
    for tok, score in token_seq:
        built += tok
        prev_pos = pos
        pos = len(built)
        if not (pos <= capture_start or prev_pos >= capture_end):
            answer_token_probs.append(score)

    if not answer_token_probs:
        # last resort: use all
        answer_token_probs = [score for _, score in token_seq]
    return answer_token_probs, answer_text


def get_uncertainty_fn(name: str):
    if name == "entropy":
        return cal_uncertainty_entropy
    if name == "avg":
        return cal_uncertainty_mean
    if name == "log_sum":
        return cal_uncertainty_log_sum
    if name == "min":
        return cal_uncertainty_min
    if name == "norm":
        return cal_uncertainty_norm
    return cal_uncertainty_entropy


def main():
    parser = argparse.ArgumentParser(description="HotpotQA case-study runner with optional scripted actions and step-wise tentative answer uncertainty.")
    parser.add_argument("--idx", type=int, required=True, help="HotpotQA question index (dev split if not changed).")
    parser.add_argument("--actions", type=str, default=None, help="Semicolon/newline separated actions (e.g., 'Search[Foo];Lookup[bar];Finish[baz]').")
    parser.add_argument("--actions_file", type=str, default=None, help="Path to a file containing actions, one per line.")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-70b-hf")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--uncertainty_method", type=str, default="entropy", choices=["entropy", "avg", "log_sum", "min", "norm"])
    parser.add_argument("--threshold", type=float, default=1.79)
    parser.add_argument("--step_scores_out", type=str, default=None, help="Optional JSONL to append per-step tentative scores.")
    parser.add_argument("--max_steps", type=int, default=7)
    parser.add_argument("--num_candidates", type=int, default=1, help="If >1, generate multiple search candidates per step and select among them.")
    parser.add_argument("--branch_scores_out", type=str, default=None, help="Optional JSONL to append per-candidate (branch) scores per step.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    model, tokenizer = build_model_and_tokenizer(args.base_model, load_in_4bit=args.load_in_4bit)
    prompter = Prompter("llama2")

    # Load prompts
    with open('./prompts/prompts.json', 'r') as f:
        prompt_dict = json.load(f)
    react_examples = prompt_dict['hotpotqa_react']

    # Build env
    env = wikienv.WikiEnv()
    env = wrappers.HotPotQAWrapper(env, split="dev")
    env = wrappers.LoggingWrapper(env)

    # Reset to specific index
    question = env.reset(idx=args.idx)
    print(f"[Question {args.idx}] {question}")

    # Scripted actions
    scripted_actions: Optional[List[str]] = None
    if args.actions_file and os.path.exists(args.actions_file):
        with open(args.actions_file, 'r') as af:
            scripted_actions = [line.strip() for line in af if line.strip()]
    elif args.actions:
        scripted_actions = parse_actions(args.actions)

    instruction_react = (
        "Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: \n"
        "(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\n"
        "(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.\n"
        "(3) Finish[answer], which returns the answer and finishes the task.\n"
        "Here are some examples.\n"
    )

    instruction_candidates = (
        "Given the question and interaction so far, propose K promising next actions for Wikipedia: both Search and Lookup are allowed.\n"
        "Use two lists on one line. Strict format: Candidates: Search=[s1; s2; ...; sK] Lookup=[l1; l2; ...; lK]\n"
        "If fewer candidates exist for either type, list fewer. Do not include any extra text.\n"
    )

    instruction_selection = (
        "You are comparing multiple candidate search results to decide which is most useful for answering the question.\n"
        "I will give you a numbered list of candidates, each with Action, Observation, TentativeAnswer, and Uncertainty.\n"
        "Pick the single best candidate ID based on relevance and lowest uncertainty.\n"
        "Respond strictly as: Select[ID]: <brief reason>\n"
    )

    traj = react_examples + question + "\n"
    uncertainty_fn = get_uncertainty_fn(args.uncertainty_method)

    def tentative_score(step_index: int, prompt_text: str):
        prefix = prompt_text + f"Thought {step_index+1}: Based on the above, I will provide an answer.\nAction {step_index+1}: Finish["
        out, probs = llama2_prompt(model, tokenizer, instruction_react, prefix, return_probs=True, max_new_tokens=32, temperature=args.temperature, top_p=args.top_p)
        answer_token_probs, answer_text = collect_span_token_scores(out, probs)
        unc = float(uncertainty_fn(answer_token_probs, 5))
        tentative_answer = answer_text.strip().rstrip('] ').lstrip('[ ')
        try:
            gold_answer = env.data[args.idx][1]
            step_em = (wrappers.normalize_answer(tentative_answer) == wrappers.normalize_answer(gold_answer))
        except Exception:
            step_em = None
        print(f"[Step {step_index}] tentative: {tentative_answer} | uncertainty={round(unc,2)} (thr={args.threshold}) | em={step_em}")
        if args.step_scores_out:
            os.makedirs(os.path.dirname(args.step_scores_out) or '.', exist_ok=True)
            with open(args.step_scores_out, 'a') as sf:
                sf.write(json.dumps({
                    "dataset": "HotpotQA",
                    "model": args.base_model,
                    "mode": "case_study_step_eval",
                    "question_idx": args.idx,
                    "step_index": step_index,
                    "tentative_answer": tentative_answer,
                    "uncertainty": round(unc, 2),
                    "threshold": args.threshold,
                    "em": step_em
                }, ensure_ascii=False) + "\n")

    def replay_and_observe(candidate_action: str, executed_actions: List[str]) -> Tuple[str, dict]:
        # Build a fresh environment and replay to current state, then perform candidate action
        env_branch = wikienv.WikiEnv()
        env_branch = wrappers.HotPotQAWrapper(env_branch, split="dev")
        env_branch = wrappers.LoggingWrapper(env_branch)
        env_branch.reset(idx=args.idx)
        for past_action in executed_actions:
            safe_env_step(env_branch, past_action)
        obs, r, done, info = safe_env_step(env_branch, candidate_action)
        return obs.replace('\\n', ''), info

    def evaluate_candidate(step_index: int, cid: int, action_type: str, argument: str, executed_actions: List[str]) -> dict:
        if action_type == "Lookup":
            action = f"Lookup[{argument}]"
        else:
            action = f"Search[{argument}]"
        obs_clean, info = replay_and_observe(action[0].lower() + action[1:], executed_actions)
        branch_traj = traj + f"Thought {step_index}: (c{cid})\nAction {step_index}: {action}\nObservation {step_index}: {obs_clean}\n"
        # Tentative answer on this branch
        out, probs = llama2_prompt(model, tokenizer, instruction_react, branch_traj + f"Thought {step_index+1}: Based on the above, I will provide an answer.\nAction {step_index+1}: Finish[", return_probs=True, max_new_tokens=32, temperature=args.temperature, top_p=args.top_p)
        answer_token_probs, answer_text = collect_span_token_scores(out, probs)
        unc = float(uncertainty_fn(answer_token_probs, 5))
        tentative_answer = answer_text.strip().rstrip('] ').lstrip('[ ')
        try:
            gold_answer = env.data[args.idx][1]
            em_val = (wrappers.normalize_answer(tentative_answer) == wrappers.normalize_answer(gold_answer))
        except Exception:
            em_val = None
        record = {
            "candidate_id": cid,
            "action_type": action_type,
            "argument": argument,
            "action": action,
            "observation": obs_clean,
            "tentative_answer": tentative_answer,
            "uncertainty": round(unc, 2),
            "em": em_val,
        }
        if args.branch_scores_out:
            os.makedirs(os.path.dirname(args.branch_scores_out) or '.', exist_ok=True)
            with open(args.branch_scores_out, 'a') as bf:
                bf.write(json.dumps({
                    "dataset": "HotpotQA",
                    "model": args.base_model,
                    "mode": "case_study_branch_eval",
                    "question_idx": args.idx,
                    "step_index": step_index,
                    **record,
                }, ensure_ascii=False) + "\n")
        return record

    def choose_best_candidate(step_index: int, candidates: List[dict]) -> int:
        # LLM-based selection; fall back to lowest uncertainty
        summary_lines = []
        for c in candidates:
            obs_snip = c["observation"][:300]
            summary_lines.append(
                f"Candidate {c['candidate_id']}:\nAction: {c['action']}\nObservation: {obs_snip}\nTentativeAnswer: {c['tentative_answer']}\nUncertainty: {c['uncertainty']}\n"
            )
        selection_input = (
            traj + f"Thought {step_index}: We evaluated {len(candidates)} candidates.\n" +
            "\n".join(summary_lines) +
            "Select the best candidate.\n"
        )
        sel_text = llama2_prompt(model, tokenizer, instruction_selection, selection_input, return_probs=False, max_new_tokens=32, temperature=args.temperature, top_p=args.top_p)
        m = re.search(r"Select\[(\d+)\]", sel_text)
        if m:
            try:
                cid = int(m.group(1))
                if any(c["candidate_id"] == cid for c in candidates):
                    return cid
            except Exception:
                pass
        # fallback: lowest uncertainty
        return min(candidates, key=lambda x: (x["uncertainty"], len(x["tentative_answer"])))["candidate_id"]

    done = False
    executed_actions: List[str] = []
    if scripted_actions is not None:
        # Execute user-provided actions
        for i, action in enumerate(scripted_actions, start=1):
            if done:
                break
            safe_action = action[0].lower() + action[1:] if action else action
            obs, r, done, info = safe_env_step(env, safe_action)
            obs_clean = obs.replace('\\n', '')
            print(f"Thought {i}: (scripted)\nAction {i}: {action}\nObservation {i}: {obs_clean}\n")
            traj += f"Thought {i}: (scripted)\nAction {i}: {action}\nObservation {i}: {obs_clean}\n"
            tentative_score(i, traj)
            if done:
                print(info)
                break
    else:
        # LLM-driven ReAct for a limited number of steps
        for i in range(1, args.max_steps + 1):
            if args.num_candidates <= 1:
                step_input = traj + f"Thought {i}:"
                thought_action = llama2_prompt(model, tokenizer, instruction_react, step_input, return_probs=False, temperature=args.temperature, top_p=args.top_p)
                try:
                    thought = thought_action.strip().split(f"\nAction {i}: ")[0]
                    action = thought_action.strip().split(f"\nAction {i}: ")[1].split("\n")[0]
                except Exception:
                    thought = thought_action.strip().split('\n')[0]
                    action_input = traj + f"Thought {i}: {thought}\nAction {i}:"
                    action = llama2_prompt(model, tokenizer, instruction_react, action_input, return_probs=False, temperature=args.temperature, top_p=args.top_p).split("\n")[0].strip()
                obs, r, done, info = safe_env_step(env, action[0].lower() + action[1:])
                executed_actions.append(action[0].lower() + action[1:])
                obs_clean = obs.replace('\\n', '')
                print(f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs_clean}\n")
                traj += f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs_clean}\n"
                tentative_score(i, traj)
                if done:
                    print(info)
                    break
            else:
                # Multi-candidate generation (Search and Lookup)
                cand_prompt = traj + f"Thought {i}: Propose {args.num_candidates} promising Search and {args.num_candidates} promising Lookup candidates.\nCandidates:"
                # Ask model under a candidate instruction to force the list format
                cand_out = llama2_prompt(model, tokenizer, instruction_candidates, cand_prompt, return_probs=False, max_new_tokens=96, temperature=args.temperature, top_p=args.top_p)
                # Parse: Candidates: Search=[a; b] Lookup=[x; y]
                m_search = re.search(r"Search=\[([^\]]*)\]", cand_out)
                m_lookup = re.search(r"Lookup=\[([^\]]*)\]", cand_out)
                if not (m_search or m_lookup):
                    # Fallback: single-step normal react
                    step_input = traj + f"Thought {i}:"
                    thought_action = llama2_prompt(model, tokenizer, instruction_react, step_input, return_probs=False, temperature=args.temperature, top_p=args.top_p)
                    try:
                        thought = thought_action.strip().split(f"\nAction {i}: ")[0]
                        action = thought_action.strip().split(f"\nAction {i}: ")[1].split("\n")[0]
                    except Exception:
                        thought = thought_action.strip().split('\n')[0]
                        action_input = traj + f"Thought {i}: {thought}\nAction {i}:"
                        action = llama2_prompt(model, tokenizer, instruction_react, action_input, return_probs=False, temperature=args.temperature, top_p=args.top_p).split("\n")[0].strip()
                    obs, r, done, info = safe_env_step(env, action[0].lower() + action[1:])
                    executed_actions.append(action[0].lower() + action[1:])
                    obs_clean = obs.replace('\\n', '')
                    print(f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs_clean}\n")
                    traj += f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs_clean}\n"
                    tentative_score(i, traj)
                    if done:
                        print(info)
                        break
                    continue
                search_entities = []
                lookup_keywords = []
                if m_search:
                    search_entities = [e.strip() for e in m_search.group(1).split(';') if e.strip()]
                if m_lookup:
                    lookup_keywords = [e.strip() for e in m_lookup.group(1).split(';') if e.strip()]
                # Limit to num_candidates for each list
                search_entities = search_entities[:args.num_candidates]
                lookup_keywords = lookup_keywords[:args.num_candidates]

                # Evaluate each candidate branch (simulated parallel via replay)
                candidate_records = []
                cid = 1
                for ent in search_entities:
                    rec = evaluate_candidate(i, cid, "Search", ent, executed_actions)
                    candidate_records.append(rec)
                    cid += 1
                for kw in lookup_keywords:
                    rec = evaluate_candidate(i, cid, "Lookup", kw, executed_actions)
                    candidate_records.append(rec)
                    cid += 1
                # Ask model to select the best candidate
                chosen_cid = choose_best_candidate(i, candidate_records)
                chosen = next(c for c in candidate_records if c["candidate_id"] == chosen_cid)
                # Adopt chosen into main trajectory and env
                chosen_label = chosen.get('argument', '')
                thought = f"I compared {len(candidate_records)} candidates and selected Candidate {chosen_cid} ({chosen['action_type']}:{chosen_label})."
                action = chosen["action"]
                obs_clean = chosen["observation"]
                print(f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs_clean}\n")
                traj += f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs_clean}\n"
                # Execute chosen action on the live env to keep it in sync
                _obs, _r, done, info = safe_env_step(env, action[0].lower() + action[1:])
                executed_actions.append(action[0].lower() + action[1:])
                tentative_score(i, traj)
                if args.branch_scores_out:
                    # Mark which candidate was chosen
                    with open(args.branch_scores_out, 'a') as bf:
                        bf.write(json.dumps({
                            "dataset": "HotpotQA",
                            "model": args.base_model,
                            "mode": "case_study_branch_choice",
                            "question_idx": args.idx,
                            "step_index": i,
                            "chosen_candidate_id": chosen_cid,
                        }, ensure_ascii=False) + "\n")
                if done:
                    print(info)
                    break


if __name__ == "__main__":
    main()


