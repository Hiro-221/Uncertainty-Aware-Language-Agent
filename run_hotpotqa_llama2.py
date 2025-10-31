import os
import re
import random
import time
import json
import torch
import wikienv, wrappers
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from utils.prompter import Prompter
from peft import PeftModel
from uncertainty_utils import *
import requests


# --------- Trace logging helpers (one JSON file per question) ---------
DEBUG_WIKI = os.environ.get("DEBUG_WIKI", "0") == "1"
# TRACE_DIR will be set after timestamp is generated


def _ensure_trace_dir(trace_dir):
    os.makedirs(trace_dir, exist_ok=True)


def _trace_path(trace_dir, question_idx):
    _ensure_trace_dir(trace_dir)
    return os.path.join(trace_dir, f"question_{question_idx}.json")


def _init_trace(dataset_name, model_name, mode_name, question_idx, question_text, gold_answer):
    return {
        "dataset": dataset_name,
        "model": model_name,
        "mode": mode_name,
        "question_idx": question_idx,
        "question": question_text,
        "gold_answer": gold_answer,
        "events": []
    }


def _log_event(trace_obj, event):
    try:
        trace_obj["events"].append(event)
    except Exception:
        pass


def _write_trace(trace_dir, question_idx, trace_obj):
    path = _trace_path(trace_dir, question_idx)
    with open(path, "w") as f:
        json.dump(trace_obj, f, ensure_ascii=False, indent=2)


# --------- Optional debugging helpers for Wikipedia suggestions ---------
def _debug_wiki_suggest(query):
    try:
        params = {
            "action": "opensearch",
            "search": query,
            "limit": 5,
            "namespace": 0,
            "format": "json",
        }
        headers = {"User-Agent": "Uncertainty-Agent/1.0 (contact: taruhiro39@gmail.com)"}
        resp = requests.get("https://en.wikipedia.org/w/api.php", params=params, timeout=8, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        # data: [search, titles[], descriptions[], urls[]]
        if isinstance(data, list) and len(data) >= 2:
            return {"status_code": resp.status_code, "titles": data[1]}
    except Exception:
        return {"status_code": None, "titles": []}
    return {"status_code": resp.status_code, "titles": []}


base_model = "meta-llama/Llama-2-70b-hf"
load_in_4bit = False # set False to use 8-bit
mode = "uala" # choose from [standard, cot, react, uala]
uncertainty_estimation_method = "entropy" # choose from [min, avg, log_sum, norm, entropy]
oracle = False # whether to use oracle in uala
save_file_name = "outputs/llama2-hotpotqa-dev-uala-nooracle.jsonl" # saved file name

# add timestamp to output file to avoid appending to existing runs
timestamp = time.strftime("%Y%m%d-%H%M%S")
base, ext = os.path.splitext(save_file_name)
save_file_name = f"{base}-{timestamp}{ext}"
# ensure output directory exists
os.makedirs(os.path.dirname(save_file_name) or ".", exist_ok=True)
step_scores_file = save_file_name.replace('.jsonl', '-step-scores.jsonl')

# Set up trace directory with timestamp
TRACE_DIR = os.path.join("traces/HotpotQA", timestamp)
os.makedirs(TRACE_DIR, exist_ok=True)

# load pre-calculated uncertainty threshold based on calibration set
if uncertainty_estimation_method == "min":
    cal_uncertainty = cal_uncertainty_min
    uncertainty_threshold = 1.79
elif uncertainty_estimation_method == "avg":
    cal_uncertainty = cal_uncertainty_mean
    uncertainty_threshold = 1.79
elif uncertainty_estimation_method == "log_sum":
    cal_uncertainty = cal_uncertainty_log_sum
    uncertainty_threshold = 10.75
elif uncertainty_estimation_method == "norm":
    cal_uncertainty = cal_uncertainty_norm
    uncertainty_threshold = 5.38
elif uncertainty_estimation_method == "entropy":
    cal_uncertainty = cal_uncertainty_entropy
    uncertainty_threshold = 1.79


tokenizer = LlamaTokenizer.from_pretrained(base_model)
prompter = Prompter("llama2")
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

def llama2_prompt(
    instruction,
    input=None,
    temperature=0,
    top_p=1,
    num_beams=1,
    do_sample=False,
    max_new_tokens=128,
    return_probs=False,
    return_top_k_candidates=False,
    top_k_size=10,
    **kwargs,
):
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        do_sample=do_sample,
        **kwargs,
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
    output = tokenizer.decode(generated_tokens[0])

    if return_probs or return_top_k_candidates:
        transition_scores = model.compute_transition_scores(
            generation_output.sequences, generation_output.scores, normalize_logits=True
        )
        prob_dicts = []
        top_k_candidates_list = []
        
        for step_idx, (tok, score) in enumerate(zip(generated_tokens[0], transition_scores[0])):
            prob_dicts.append({tokenizer.decode(tok):score.cpu().tolist()})
            
            if return_top_k_candidates:
                # Get the logits for this step and convert to probabilities
                step_scores = generation_output.scores[step_idx][0]  # shape: [vocab_size]
                probs = torch.nn.functional.softmax(step_scores, dim=-1)
                
                # Get top-k tokens and their probabilities
                top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k_size, probs.shape[0]))
                
                candidates = []
                for prob, idx in zip(top_k_probs, top_k_indices):
                    token_str = tokenizer.decode([idx.item()])
                    candidates.append({
                        "token": token_str,
                        "token_id": idx.item(),
                        "probability": prob.item()
                    })
                
                top_k_candidates_list.append({
                    "position": step_idx,
                    "selected_token": tokenizer.decode(tok),
                    "selected_token_id": tok.item(),
                    "candidates": candidates
                })

        if return_top_k_candidates:
            return output, prob_dicts, top_k_candidates_list
        else:
            return output, prob_dicts

    else:
        return output

env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)

def step(env, action):
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
            if DEBUG_WIKI:
                print(f"[DEBUG_WIKI] Timeout on action={action!r}, retry={attempts}")
            # No trace object here; caller will attach retry info per step

prompt_file = './prompts/prompts.json'
with open(prompt_file, 'r') as f:
    prompt_dict = json.load(f)

# standard prompt
hotpotqa_standard_examples = prompt_dict['hotpotqa_standard']
instruction_standard = """Answer the question:\n"""

# cot prompt
hotpotqa_cot_examples = prompt_dict['hotpotqa_cot']
instruction_cot = """Solve a question answering task. Your task is to generate Thought and Answer where a Thought can reason about the current situation by thinking step by step.
Here are some examples.
"""

# react prompt
hotpotqa_react_examples = prompt_dict['hotpotqa_react']
instruction_react = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""


def standard(idx=None, instruction=instruction_standard, prompt=hotpotqa_standard_examples, to_print=True, trace=None):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)
    prompt += question + "\n"
    full_input = prompt + "Answer:"
    answer_raw, probs = llama2_prompt(instruction, full_input, return_probs=True)
    if trace is not None:
        full_prompt_text = prompter.generate_prompt(instruction, full_input)
        _log_event(trace, {
            "type": "llm_call",
            "stage": "standard",
            "instruction": instruction,
            "input": full_input,
            "prompt": full_prompt_text,
            "output": answer_raw,
            "probs": probs
        })
    answer = answer_raw.split("\n")[0].strip()
    if to_print:
        print("Answer:", answer)

    token_probs = []
    for d in probs:
        d = {k.replace('<0x0A>', '\n'): v for k, v in d.items()}
        key = [key for key in d.keys()][0]
        if key in answer:
            token_probs.append({key:d[key]})
        else:
            break
    return answer, token_probs

def cot(idx=None, instruction=instruction_cot, prompt=hotpotqa_cot_examples, to_print=True, trace=None):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)
    prompt += question + "\n"
    full_input = prompt + "Thought:"
    answer_raw, probs = llama2_prompt(instruction, full_input, return_probs=True)
    if trace is not None:
        full_prompt_text = prompter.generate_prompt(instruction, full_input)
        _log_event(trace, {
            "type": "llm_call",
            "stage": "cot",
            "instruction": instruction,
            "input": full_input,
            "prompt": full_prompt_text,
            "output": answer_raw,
            "probs": probs
        })
    answer = answer_raw.split("\nQuestion:")[0].strip()
    if to_print:
        print("Thought:", answer)

    token_probs = []
    for d in probs:
        d = {k.replace('<0x0A>', '\n'): v for k, v in d.items()}
        key = [key for key in d.keys()][0]
        if key in answer:
            token_probs.append({key:d[key]})
        else:
            break
    return answer, token_probs

def react(idx=None, instruction=instruction_react, prompt=hotpotqa_react_examples, to_print=True, trace=None, per_step_jsonl_path=None):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    react_probs = []
    for i in range(1, 8):
        n_calls += 1
        step_input = prompt + f"Thought {i}:"
        thought_action, thought_action_probs = llama2_prompt(instruction, step_input, return_probs=True)
        if DEBUG_WIKI:
            print(f"[DEBUG_WIKI] LLM raw ThoughtAction {i}: {repr(thought_action)}")
        if trace is not None:
            _log_event(trace, {
                "type": "llm_call",
                "stage": f"react_step_{i}",
                "instruction": instruction,
                "input": step_input,
                "prompt": prompter.generate_prompt(instruction, step_input),
                "output": thought_action,
                "probs": thought_action_probs
            })
        react_probs.append(thought_action_probs)
        try:
            thought = thought_action.strip().split(f"\nAction {i}: ")[0]
            action = thought_action.strip().split(f"\nAction {i}: ")[1].split("\n")[0]
        except:
            print('ohh...', thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action_input = prompt + f"Thought {i}: {thought}\nAction {i}:"
            action, action_probs= llama2_prompt(instruction, action_input, return_probs=True)
            action = action.split("\n")[0].strip()
            react_probs.append(action_probs)
            if trace is not None:
                _log_event(trace, {
                    "type": "llm_call",
                    "stage": f"react_step_{i}_action",
                    "instruction": instruction,
                    "input": action_input,
                    "prompt": prompter.generate_prompt(instruction, action_input),
                    "output": action,
                    "probs": action_probs
                })
        if DEBUG_WIKI:
            print(f"[DEBUG_WIKI] Parsed Thought {i}: {thought}")
        step_start = time.time()
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        if DEBUG_WIKI:
            print(f"[DEBUG_WIKI] Action {i}: {action}")
            print(f"[DEBUG_WIKI] Raw Observation {i}: {repr(obs)}")
            print(f"[DEBUG_WIKI] Info {i}: {info}")
            # Suggest candidates if search returns empty
            if obs == "" and action.lower().startswith("search[") and action.endswith("]"):
                q = action[action.find("[") + 1: action.rfind("]")]
                suggestions = _debug_wiki_suggest(q)
                titles = suggestions.get("titles", [])
                if titles:
                    print(f"[DEBUG_WIKI] Suggestions for {q!r}: {titles}")
                else:
                    print(f"[DEBUG_WIKI] No suggestions for {q!r}")
        raw_obs = obs
        obs = obs.replace('\\n', '')

        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        if to_print:
            print(step_str)
        # After each observation, attempt a tentative answer and compute its uncertainty (does not alter main prompt)
        if per_step_jsonl_path is not None:
            try:
                tentative_prefix = prompt + f"Thought {i+1}: Based on the above, I will provide an answer.\nAction {i+1}: Finish["
                # Generate full prompt for recording
                full_prompt_for_tentative = prompter.generate_prompt(instruction, tentative_prefix)
                # Get output with detailed token candidates
                tentative_out, tentative_probs, top_k_candidates = llama2_prompt(
                    instruction, 
                    tentative_prefix, 
                    return_probs=True, 
                    return_top_k_candidates=True,
                    top_k_size=10,
                    max_new_tokens=32
                )
                # Extract token scores for the span inside brackets by aligning token strings to generated text
                # Build ordered list of (token_str, score)
                token_seq = []
                for d in tentative_probs:
                    # each d is a single {token: score}
                    for k, v in d.items():
                        token_seq.append((k, v))

                gen_text = tentative_out
                start_idx = gen_text.find('[')
                end_idx = gen_text.find(']', start_idx + 1) if start_idx != -1 else -1

                # Fallback: if no closing bracket, use up to first newline, else full text
                if start_idx != -1 and end_idx != -1:
                    answer_text = gen_text[start_idx + 1:end_idx]
                    capture_start = start_idx + 1
                    capture_end = end_idx
                else:
                    nl = gen_text.find('\n')
                    if nl == -1:
                        answer_text = gen_text.strip()
                        capture_start = 0
                        capture_end = len(gen_text)
                    else:
                        answer_text = gen_text[:nl].strip()
                        capture_start = 0
                        capture_end = nl

                # Align tokens to text spans
                answer_token_probs = []
                built = ""
                pos = 0
                for tok, score in token_seq:
                    built += tok
                    prev_pos = pos
                    pos = len(built)
                    # overlap check with [capture_start, capture_end)
                    if not (pos <= capture_start or prev_pos >= capture_end):
                        answer_token_probs.append(score)

                if not answer_token_probs:
                    # as a last resort, use all token scores
                    for _, score in token_seq:
                        answer_token_probs.append(score)
                tentative_unc = cal_uncertainty(answer_token_probs, 5)
                tentative_answer = answer_text.strip()
                # compute step EM if possible
                try:
                    gold_answer = env.data[idx][1]
                    step_em = (wrappers.normalize_answer(tentative_answer) == wrappers.normalize_answer(gold_answer))
                except Exception:
                    step_em = None
                # print and persist
                print(f"Step {i} tentative uncertainty: {round(tentative_unc,2)}")
                # capture search keyword if this step executed a Search action
                search_keyword = None
                try:
                    if isinstance(action, str) and action.lower().startswith("search[") and action.endswith("]"):
                        search_keyword = action[action.find("[") + 1: action.rfind("]")]
                except Exception:
                    search_keyword = None
                with open(per_step_jsonl_path, 'a') as sf:
                    sf.write(json.dumps({
                        "dataset": "HotpotQA",
                        "model": base_model,
                        "mode": "uala_step_eval",
                        "question_idx": idx,
                        "step_index": i,
                        "thought": thought,
                        "search_keyword": search_keyword,
                        "tentative_answer": tentative_answer,
                        "uncertainty": round(tentative_unc, 2),
                        "threshold": uncertainty_threshold,
                        "em": step_em
                    }, ensure_ascii=False) + "\n")
                if trace is not None:
                    _log_event(trace, {
                        "type": "tentative_step_answer",
                        "index": i,
                        "tentative_answer": tentative_answer,
                        "uncertainty": round(tentative_unc, 2),
                        "full_prompt": full_prompt_for_tentative,
                        "generated_output": tentative_out,
                        "top_k_candidates": top_k_candidates
                    })
            except Exception as e:
                if to_print:
                    print(f"[WARN] Tentative answer at step {i} failed: {e}")
        if trace is not None:
            # Extract similar titles structurally when mismatch message appears
            similar_titles = None
            if isinstance(raw_obs, str) and raw_obs.startswith("Could not find ") and "Similar:" in raw_obs:
                try:
                    # raw format: Could not find X. Similar: ['A', 'B', ...].
                    after = raw_obs.split("Similar:", 1)[1].strip()
                    # ensure we only parse the list part
                    list_text = after
                    similar_titles = ast.literal_eval(list_text)
                except Exception:
                    similar_titles = None

            _log_event(trace, {
                "type": "react_step",
                "index": i,
                "thought": thought,
                "action": action,
                "observation": obs,
                "raw_observation": raw_obs,
                "info": info,
                "similar_titles": similar_titles
            })
            if DEBUG_WIKI and raw_obs == "" and action.lower().startswith("search[") and action.endswith("]"):
                q = action[action.find("[") + 1: action.rfind("]")]
                sug = _debug_wiki_suggest(q)
                _log_event(trace, {
                    "type": "debug_wiki_suggestions",
                    "query": q,
                    "status_code": sug.get("status_code"),
                    "suggestions": sug.get("titles", [])
                })
        if done:
            break
    if not done:
        obs, r, done, info = step(env, "finish[]")

    if to_print:
        print(info, '\n')
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return info, react_probs


idxs = list(range(7405))
random.Random(233).shuffle(idxs)

evals = []
old_time = time.time()

num_tool_call_instance = 0
num_instance = 0
num_correct = 0
num_tool_calls = 0
num_backoff = 0
num_ask_human = 0

with open(save_file_name,"a") as output_file:
    for i in tqdm(idxs[:500]):
        question = env.reset(idx=i)
        gold_answer = env.data[i][1]
        num_instance += 1
        # initialize per-question trace
        trace = _init_trace("HotpotQA", base_model, mode, i, question, gold_answer)

        if mode == "standard":
            predicted_answer, _ = standard(i, to_print=True, trace=trace)
            print('-----------')
            em = (wrappers.normalize_answer(predicted_answer) == wrappers.normalize_answer(gold_answer))
            f1 = wrappers.f1_score(wrappers.normalize_answer(predicted_answer), wrappers.normalize_answer(gold_answer))[0]
            standard_final_output = {"answer": predicted_answer, "gt_answer": gold_answer, "question_idx": i, "em": em, "f1": f1, "traj": question + '\nAnswer:' + predicted_answer}
            if standard_final_output["em"]:
                num_correct += 1
            output_file.write(json.dumps(standard_final_output, ensure_ascii=False) + '\n')
            trace["traj"] = standard_final_output["traj"]
            trace["summary"] = {"final_answer": predicted_answer, "em": em, "f1": f1, "mode": "standard"}
            _write_trace(TRACE_DIR, i, trace)

        elif mode == "cot":
            cot_output, _ = cot(i, to_print=True, trace=trace)
            print('-----------')
            try:
                predicted_answer = cot_output.split('Answer:')[1].strip()
            except:
                try:
                    predicted_answer = cot_output.split("the answer is ")[1].strip()
                except:
                    predicted_answer = ""
            em = (wrappers.normalize_answer(predicted_answer) == wrappers.normalize_answer(gold_answer))
            f1 = wrappers.f1_score(wrappers.normalize_answer(predicted_answer), wrappers.normalize_answer(gold_answer))[0]
            cot_final_output = {"answer": predicted_answer, "gt_answer": gold_answer, "question_idx": i, "em": em, "f1": f1, "traj": question + '\nThought:' + cot_output}
            if cot_final_output["em"]:
                num_correct += 1
            output_file.write(json.dumps(cot_final_output, ensure_ascii=False) + '\n')
            trace["traj"] = cot_final_output["traj"]
            trace["summary"] = {"final_answer": predicted_answer, "em": em, "f1": f1, "mode": "cot"}
            _write_trace(TRACE_DIR, i, trace)

        elif mode == "react":
            info, react_probs = react(i, to_print=True, trace=trace)
            evals.append(info['em'])
            print(sum(evals), len(evals), sum(evals) / len(evals), (time.time() - old_time) / len(evals))
            print('-----------')
            # compute uncertainty of the ReAct final answer
            answer_probs = []
            for list in react_probs:
                for key in list:
                    finish = False
                    if "Fin" in key:
                        answer_probs = []
                        idx = list.index(key)
                        for i in range(idx+2,len(list)):
                            for key, value in list[i].items():
                                if key == '[':
                                    continue
                                if key == ']':
                                    finish = True
                                    break
                                answer_probs.append(value)
                            if finish:
                                break
            if not answer_probs:
                answer_probs = [0.0]
            react_uncertainty = cal_uncertainty(answer_probs, 5)
            info["react_uncertainty"] = round(react_uncertainty, 2)
            print(f"ReAct answer uncertainty: {round(react_uncertainty, 2)} (threshold: {uncertainty_threshold})")
            _log_event(trace, {
                "type": "measure_uncertainty",
                "phase": "react_answer",
                "uncertainty": round(react_uncertainty, 2),
                "threshold": uncertainty_threshold,
                "answer_probs_len": len(answer_probs)
            })
            info["traj"] = info["traj"].split(hotpotqa_react_examples)[1].strip()
            num_tool_calls += info["n_calls"]
            if info["em"]:
                num_correct += 1
            output_file.write(json.dumps(info, ensure_ascii=False) + '\n')
            trace["traj"] = info["traj"]
            trace["summary"] = {"final_answer": info.get("answer"), "em": info.get("em"), "f1": info.get("f1"), "mode": "react", "n_calls": info.get("n_calls"), "n_badcalls": info.get("n_badcalls"), "react_uncertainty": info.get("react_uncertainty")}
            _write_trace(TRACE_DIR, i, trace)

        elif mode == "uala":
            print('-----------CoT-----------')
            cot_output, probs = cot(i, to_print=True, trace=trace)
            try:
                predicted_answer = cot_output.split('Answer:')[1].strip()
            except:
                try:
                    predicted_answer = cot_output.split("the answer is ")[1].strip()
                except:
                    predicted_answer = ""

            em = (wrappers.normalize_answer(predicted_answer) == wrappers.normalize_answer(gold_answer))
            f1 = wrappers.f1_score(wrappers.normalize_answer(predicted_answer), wrappers.normalize_answer(gold_answer))[0]
            cot_final_output = {"steps": 2, "answer": predicted_answer, "gt_answer": gold_answer, "question_idx": i, "reward": em, "em": em, "f1": f1, "n_calls": 0, "n_badcalls": 0, "traj": question + '\nThought 1:' + cot_output + f'\nAction 1: MeasureUncertainty [{predicted_answer}]'}
            
            # extract answer token logprobs for hotpotqa
            answer_probs = []
            idx = -1
            for d in probs:
                for key in d.keys():
                    if key == "Answer":
                        idx = probs.index(d)
            for d in probs[idx+2:-1]:
                for value in d.values():
                    answer_probs.append(value)      
            if not answer_probs:
                for d in probs[:-1]:
                    for value in d.values():
                        answer_probs.append(value)

            # calculate uncertainty
            uncertainty = cal_uncertainty(answer_probs, 5)
            _log_event(trace, {
                "type": "measure_uncertainty",
                "phase": "cot_answer",
                "uncertainty": round(uncertainty, 2),
                "threshold": uncertainty_threshold,
                "answer_probs_len": len(answer_probs)
            })
            cot_final_output["cot_uncertainty"] = round(uncertainty, 2)

            # make tool use
            if uncertainty > uncertainty_threshold:
                print(f'-----------Answer’s uncertainty {round(uncertainty,2)}, which falls outside the acceptable uncertainty threshold of {uncertainty_threshold}, I need to use an external tool to solve the question.-----------')
                num_tool_call_instance += 1
                print('-----------ReAct-----------')
                _log_event(trace, {"type": "tool_activation", "tool": "ReAct"})
                info, react_probs = react(i, to_print=True, trace=trace, per_step_jsonl_path=step_scores_file)
                predicted_react_answer = info["answer"]
                cot_final_output["traj"] += f"\nObservation 1: Answer’s uncertainty is {round(uncertainty,2)}, which falls outside the acceptable threshold of {uncertainty_threshold}.\nThought 2: Based on the uncertainty, I need to use an external tool to solve the question.\nAction 2: Activate Tool.\n"
                
                def reformat(text):
                    return f"{text.group(1)} {int(text.group(2)) + 2}"

                react_traj = info["traj"].split(hotpotqa_react_examples + question + "\n")[1].strip()
                reformat_react_traj = re.compile(r'(Thought|Observation|Action) (\d+)').sub(reformat, react_traj)

                last_index = int(re.findall(r'(Thought|Observation|Action)\s*(\d+)', reformat_react_traj)[-1][-1])

                info["traj"] = cot_final_output["traj"] + reformat_react_traj 
                info["steps"] += cot_final_output["steps"]
                num_tool_calls += info["n_calls"]

                if info["answer"]:
                    info["traj"] += f"\nThought {str(last_index+1)}: Let me check the uncertainty of the returned answer.\nAction {str(last_index+1)}: MeasureUncertainty [{predicted_react_answer}]"
                    info["steps"] += 1

                    # extract the answer token probs
                    answer_probs = []
                    for list in react_probs:
                        for key in list:
                            finish = False
                            if "Fin" in key:
                                answer_probs = []
                                idx = list.index(key)
                                for i in range(idx+2,len(list)):
                                    for key, value in list[i].items():
                                        if key == '[':
                                            continue
                                        if key == ']':
                                            finish = True
                                            break
                                        answer_probs.append(value)
                                    if finish:
                                        break
                    if not answer_probs:
                        answer_probs = [0.0]

                    react_uncertainty = cal_uncertainty(answer_probs, 5)
                    info["react_uncertainty"] = round(react_uncertainty, 2)
                    info["cot_uncertainty"] = round(uncertainty, 2)
                    _log_event(trace, {
                        "type": "measure_uncertainty",
                        "phase": "react_answer",
                        "uncertainty": round(react_uncertainty, 2),
                        "threshold": uncertainty_threshold,
                        "answer_probs_len": len(answer_probs)
                    })
                    print(f"ReAct answer uncertainty: {round(react_uncertainty, 2)} (threshold: {uncertainty_threshold})")
                    if react_uncertainty > uncertainty_threshold:
                        info["steps"] += 1
                        if oracle:
                            print(f"-----------Answer’s uncertainty is {round(react_uncertainty,2)}, which falls outside the acceptable threshold of {uncertainty_threshold}, ask a human for help.-----------")
                            info["traj"] += f"\nObservation {str(last_index+2)}: Answer’s uncertainty is {round(react_uncertainty,2)}, which falls outside the acceptable threshold of {uncertainty_threshold}.\nThought {str(last_index+2)}: Based on the uncertainty, I need to ask a human for help.\nAction {str(last_index+2)}: Ask Human.\nObservation {str(last_index+2)}: {gold_answer}\nAnswer: {gold_answer}"
                            info["answer"] = gold_answer
                            info["reward"] = True
                            info["em"] = True
                            info["f1"] = 1.0
                            num_ask_human += 1
                            _log_event(trace, {"type": "ask_human", "gold_answer": gold_answer})
                        else:
                            print(f"-----------Answer’s uncertainty is {round(react_uncertainty,2)}, which falls outside the acceptable threshold of {uncertainty_threshold}, for simplicity, answer is still kept.-----------")
                            info["traj"] += f"\nObservation {str(last_index+2)}: Answer’s uncertainty is {round(react_uncertainty,2)}, which falls outside the acceptable threshold of {uncertainty_threshold}.\nThought {str(last_index+2)}: For simplicity, answer is still kept.\nAction {str(last_index+2)}: Keep Answer.\nAnswer: {predicted_react_answer}"
                            _log_event(trace, {"type": "keep_answer", "phase": "react_answer_high_uncertainty", "answer": predicted_react_answer})
                            
                    else:
                        print(f"-----------Answer’s uncertainty is {round(react_uncertainty,2)}, which falls within the acceptable threshold of {uncertainty_threshold}, answer is kept.-----------")
                        info["traj"] += f"\nObservation {str(last_index+2)}: Answer’s uncertainty is {round(react_uncertainty,2)}, which falls within the acceptable threshold of {uncertainty_threshold}.\nThought {str(last_index+2)}: Based on the uncertainty, answer is kept.\nAction {str(last_index+2)}: Keep Answer.\nAnswer: {predicted_react_answer}"
                        info["steps"] += 1
                        _log_event(trace, {"type": "keep_answer", "phase": "react_answer", "answer": predicted_react_answer})
                    
                else:
                    print("-----------Returned tool-use answer is invalid, I need to use the backoff answer.-----------")
                    use_backoff_traj = f"\nThought {str(last_index+1)}: Returned tool-use answer is invalid, I need to use the backoff answer.\nAction {str(last_index+1)}: Use Backoff Answer.\nAnswer: {predicted_answer}"
                    info["traj"] += use_backoff_traj
                    num_backoff += 1
                    info["answer"] = cot_final_output["answer"]
                    info["reward"] = cot_final_output["reward"]
                    info["em"] = cot_final_output["em"]
                    info["f1"] = cot_final_output["f1"]
                    info["cot_uncertainty"] = round(uncertainty, 2)
                    _log_event(trace, {"type": "use_backoff_answer", "answer": predicted_answer})

                if info["em"]:
                    num_correct += 1
                output_file.write(json.dumps(info, ensure_ascii=False) + '\n')
                trace["traj"] = info["traj"]
                trace["summary"] = {"final_answer": info.get("answer"), "em": info.get("em"), "f1": info.get("f1"), "mode": "uala", "n_calls": info.get("n_calls"), "n_badcalls": info.get("n_badcalls"), "cot_uncertainty": info.get("cot_uncertainty"), "react_uncertainty": info.get("react_uncertainty")}
                _write_trace(TRACE_DIR, i, trace)
            else:
                print(f'-----------Answer\'s uncertainty is {round(uncertainty,2)}, which falls within the acceptable threshold of {uncertainty_threshold}, answer is kept.-----------')
                cot_final_output["traj"] += f"\nObservation 1: Answer\'s uncertainty is {round(uncertainty,2)}, which falls within the acceptable threshold of {uncertainty_threshold}.\nThought 2: Based on the uncertainty, answer is kept.\nAction 2: Keep Answer.\nAnswer: {predicted_answer}"
                if cot_final_output["em"]:
                    num_correct += 1
                output_file.write(json.dumps(cot_final_output, ensure_ascii=False) + '\n')
                trace["traj"] = cot_final_output["traj"]
                trace["summary"] = {"final_answer": cot_final_output.get("answer"), "em": cot_final_output.get("em"), "f1": cot_final_output.get("f1"), "mode": "uala_cot_only", "cot_uncertainty": cot_final_output.get("cot_uncertainty")}
                _write_trace(TRACE_DIR, i, trace)
