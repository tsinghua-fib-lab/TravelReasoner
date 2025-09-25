#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process completion outputs into a single CSV.

- Reads JSON files under a folder (default: outputs/)
- Extracts profile fields, stay/travel times, and parsed table content
- Aggregates per file into one row
- Writes CSV to outputs_processed/outputs_processed.csv by default

功能不变，仅做了代码风格与可读性优化（排版、注释、文档、轻量类型注解）。
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from typing import Iterable, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTIVITY_LIST: List[str] = [
    "Regular home activities (chores, sleep)",
    "Work from home (paid)",
    "Work",
    "Work-related meeting / trip",
    "Volunteer activities (not paid)",
    "Drop off / pick up someone",
    "Change type of transportation",
    "Attend school as a student",
    "Attend child care",
    "Attend adult care",
    "Buy goods (groceries, clothes, appliances, gas)",
    "Buy services (dry cleaners, banking, service a car, etc)",
    "Buy meals (go out for a meal, snack, carry-out)",
    "Other general errands (post office, library)",
    "Recreational activities (visit parks, movies, bars, etc)",
    "Exercise (go for a jog, walk, walk the dog, go to the gym, etc)",
    "Visit friends or relatives",
    "Health care visit (medical, dental, therapy)",
    "Religious or other community activities",
    "Something else",
]

# ---------------------------------------------------------------------------
# Time utilities
# ---------------------------------------------------------------------------


def _to_hhmm(tstr: str) -> str:
    """
    Convert "h:mm AM/PM" to "HHMM" 24-hour compact time.

    Examples
    --------
    '9:30 AM'  -> '0930'
    '12:00 PM' -> '1200'
    '12:00 AM' -> '0000'
    '11:59 PM' -> '2359'

    容错：
    - 若字符串已是 3~4 位数字或空字符串，则原样/提取后返回（兜底）。
    """
    tstr = tstr.strip()
    m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*([AP]M)\s*$", tstr, re.I)
    if not m:
        # 兜底：如果已是 3~4 位数字或空，原样返回（仅取数字部分）
        n = re.match(r"^\s*(\d{3,4})\s*$", tstr)
        return n.group(1) if n else tstr

    hh = int(m.group(1))
    mm = int(m.group(2))
    ampm = m.group(3).upper()

    if ampm == "AM":
        if hh == 12:
            hh = 0
    else:  # PM
        if hh != 12:
            hh += 12
    return f"{hh:02d}{mm:02d}"


def norm_time_str(x: int | str) -> str:
    """Normalize int/str to 4-digit 'HHMM' string."""
    return f"{int(x):04d}"


# ---------------------------------------------------------------------------
# Profile parsing
# ---------------------------------------------------------------------------


def extract_profile_fields(text: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    从长文本 text 中提取：
      - Age -> int (如 77)
      - Gender -> str (如 'male'；统一为小写)
      - Household annual income per capita -> str, 形如 '$112500'（去掉逗号，确保有 $）

    返回最后一个匹配项。
    若未匹配到，返回 (None, None, None)。
    """
    if not text:
        return None, None, None

    # 多行匹配 + 忽略大小写；允许中英文冒号
    age_matches = re.findall(r"(?im)^\s*Age\s*[:：]\s*(\d+)\s*$", text)
    gender_matches = re.findall(r"(?im)^\s*Gender\s*[:：]\s*([A-Za-z]+)\s*$", text)
    income_matches = re.findall(
        r"(?im)^\s*Household\s+annual\s+income\s+per\s+capita\s*[:：]\s*(\$?\s*[\d,]+)\s*$",
        text,
    )

    # 获取最后一个匹配项，若存在
    age = int(age_matches[-1]) if age_matches else None
    gender = gender_matches[-1].strip().lower() if gender_matches else None

    income: Optional[str] = None
    if income_matches:
        raw = income_matches[-1]
        cleaned = raw.replace(" ", "").replace(",", "")
        if not cleaned.startswith("$"):
            cleaned = "$" + cleaned
        income = cleaned

    return age, gender, income


# ---------------------------------------------------------------------------
# Time block conversion
# ---------------------------------------------------------------------------


def travel_to_stay(travel_times: Iterable[Iterable[int | str]]) -> List[List[str]]:
    """
    把 travel_times 转为 stay_times。

    Parameters
    ----------
    travel_times : [[1100,1140], [1145,1230], ...] （int 或 str）

    Returns
    -------
    [['0000','1100'], ['1140','1145'], ..., ['1510','2400']]
    """
    if not travel_times:
        return [["0000", "2400"]]

    def norm(x: int | str) -> str:
        return f"{int(x):04d}"

    tt = list(travel_times)
    stay_times: List[List[str]] = []

    # 第一个停留：从午夜到第一段出行开始
    first_start = norm(tt[0][0])
    stay_times.append(["0000", first_start])

    # 中间停留：相邻两段出行之间
    for i in range(len(tt) - 1):
        prev_end = norm(tt[i][1])
        next_start = norm(tt[i + 1][0])
        stay_times.append([prev_end, next_start])

    # 最后一个停留：最后一段结束到午夜
    last_end = norm(tt[-1][1])
    stay_times.append([last_end, "2400"])

    return stay_times


def stay_to_travel(stay_times: Iterable[Iterable[str]]) -> List[List[str]]:
    """
    将停留时段转换为出行时段。

    Parameters
    ----------
    stay_times : [['0000','0830'], ['0930','1200'], ...]

    Returns
    -------
    [['0830','0930'], ['1200','1300'], ...]
    """
    st = list(stay_times)
    travel_times: List[List[str]] = []
    for i in range(len(st) - 1):
        dep = st[i][1]    # 当前停留结束
        arr = st[i + 1][0]  # 下一停留开始
        travel_times.append([dep, arr])
    return travel_times


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _to_int_list_maybe(x) -> List[int] | str | list:
    """
    尝试把类似 "[1, 6, 11]" 的字符串解析为 List[int]。
    - 若已是 list，则转成 int 列表（容错：元素可能是 str）
    - 若是 str，则用 ast.literal_eval 解析，并转成 int 列表
    - 解析失败则原样返回（不抛错）
    """
    try:
        if isinstance(x, list):
            return [int(v) for v in x]
        if isinstance(x, str):
            parsed = ast.literal_eval(x)  # 安全解析
            if isinstance(parsed, list):
                return [int(v) for v in parsed]
    except Exception:
        pass
    return x



# ---------------------------------------------------------------------------
# get_activity_code
# ---------------------------------------------------------------------------
from difflib import get_close_matches

def get_activity_code(loc_type_raw: str) -> int:
    # 尝试从 loc_type_raw 提取前缀数字
    mcode = re.match(r"^\s*(\d+)", loc_type_raw)
    if mcode:
        # 如果提取到数字，返回该数字
        return int(mcode.group(1))
    
    # 如果没有提取到数字，进行模糊匹配
    # 首先尝试完全匹配
    for i, activity in enumerate(ACTIVITY_LIST, start=1):
        if activity.lower() == loc_type_raw.lower():
            if activity.lower() == "something else":
                return 97
            return i
    
    # 如果没有完全匹配，使用模糊匹配
    # 创建一个简化的活动列表用于模糊匹配（去除括号内的描述）
    simple_activity_list = []
    for activity in ACTIVITY_LIST:
        # 提取括号前的主要描述
        simple_activity = re.split(r'\(|\)', activity)[0].strip().lower()
        simple_activity_list.append(simple_activity)
    
    # 在简化列表中进行模糊匹配
    matches = get_close_matches(loc_type_raw.lower(), simple_activity_list, n=1, cutoff=0.6)
    if matches:
        # 找到匹配项在原始列表中的索引
        matched_index = simple_activity_list.index(matches[0])
        return matched_index + 1  # 因为索引从0开始，但活动代码从1开始
    
    # 如果模糊匹配也没有找到，检查是否包含关键词
    keywords_mapping = {
        "home": 1,
        "work": 3,  # 默认到"Work"而不是"Work from home"
        "meeting": 4,
        "trip": 4,
        "volunteer": 5,
        "drop": 6,
        "pick up": 6,
        "transportation": 7,
        "school": 8,
        "child care": 9,
        "adult care": 10,
        "buy": 11,  # 默认到"Buy goods"
        "goods": 11,
        "services": 12,
        "meal": 13,
        "eat": 13,
        "restaurant": 13,
        "errand": 14,
        "recreational": 15,
        "exercise": 16,
        "jog": 16,
        "walk": 16,
        "gym": 16,
        "visit": 17,
        "friend": 17,
        "relative": 17,
        "health": 18,
        "care": 18,
        "medical": 18,
        "religious": 19,
        "community": 19,
    }
    
    for keyword, code in keywords_mapping.items():
        if keyword in loc_type_raw.lower():
            return code
    
    # 如果没有任何匹配，返回默认值 97
    return 97


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def process_outputs_completion(
    folder_loc: str = "outputs/",
    fix_missing: bool = False,
):
    """
    扫描 folder_loc 目录下的 JSON 文件，解析并聚合生成一张 DataFrame 所需的数据列表。

    Parameters
    ----------
    folder_loc : str
        输入文件夹，内含若干 .json
    fix_missing : bool
        若表格不完整时，是否自动补尾（补一段 Home -> 1，至 23:59 PM）

    Returns
    -------
    List[dict]
        每个 JSON 文件对应一行聚合后的字典数据。
    """
    files = os.listdir(folder_loc)
    data = []
    counter = 0
    incomplete_generations = 0
    table_not_found = 0

    for fname in files:
        if not fname.endswith(".json"):
            continue

        f_name_no_ext = fname.split(".")[0]
        with open(os.path.join(folder_loc, fname), "r", encoding="utf-8") as f:
            answers = json.load(f)

        # 基础 info：复制 answers，去掉大字段
        info = answers.copy()
        for a in ["ans_output", "input", "travel_times"]:
            if a in info:
                del info[a]

        # 原始 JSON 里的 loc_type → 改名为 loc_type_actual，避免与表格列冲突
        if "loc_type" in info:
            info["loc_type_actual"] = info.pop("loc_type")
            try:
                info["loc_type_actual"] = ast.literal_eval(info["loc_type_actual"])
            except Exception:
                # 保留原值（若无法解析）
                pass

        if "loc_type_actual" in info:
            info["loc_type_actual"] = _to_int_list_maybe(info["loc_type_actual"])

        # 从 input 中提 Age/Gender/Income，回填到 info（若已存在 Income，就不覆盖）
        age, gender, income = extract_profile_fields(answers.get("input", ""))
        if age is not None:
            info["Age"] = age
        if gender is not None:
            info["Gender"] = gender
        if income is not None and "Income" not in info:
            info["Income"] = income

        # 规范 travel_times（若存在）
        raw_tt = answers.get("travel_times", [])
        if raw_tt:
            travel_times_norm = [[norm_time_str(s), norm_time_str(e)] for s, e in raw_tt]
        else:
            travel_times_norm = []
        info["travel_time_actual"] = travel_times_norm

        # —— 解析 ans_output 的表格 —— #
        ans_output = answers.get("ans_output", "")
        ds = [d.strip() for d in ans_output.split("\n")]

        # 找到表头分隔线所在行（如：| ---- | ---- | ---- | ---- |）
        header_idx = None
        pattern = re.compile(r"\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|", re.I)
        for i, line in enumerate(ds):
            if pattern.match(line):
                header_idx = i
                break

        if header_idx is None:
            table_not_found += 1
            continue

        # 收集本文件（同一 key）的聚合结果
        loc_type_gen: List[int] = []
        stay_time_gen: List[List[str]] = []

        a = header_idx
        incomplete_generation = False

        # 只收集必要字段并聚合，不逐行输出临时列
        while True:
            if a + 1 >= len(ds):
                break
            if ds[a + 1] == "":
                break

            t = [x.strip() for x in ds[a + 1].split("|")]
            # 预期形如: ['', Place, Arrival, Departure, LocType, '']
            if len(t) == 6 and t[4] != "[Location Type]":
                # place_name = t[1]  # 当前逻辑并未输出这些列，保留注释
                arrival = t[2]
                departure = t[3]
                loc_type_raw = t[4]
                
                code= get_activity_code(loc_type_raw)
                # loc_type 取前缀数字，如 "97: Something else" -> 97
                # mcode = re.match(r"^\s*(\d+)", loc_type_raw)
                # if mcode:
                #     code = int(mcode.group(1))
                # else:
                #     # 若未匹配到数字，给个兜底 97
                #     code = 97

                loc_type_gen.append(code)
                stay_time_gen.append([_to_hhmm(arrival), _to_hhmm(departure)])
                a += 1
            else:
                # 不完整生成（表格提前结束/格式异常）
                incomplete_generation = True
                if fix_missing:
                    # 自动补尾：Home，从上一段的离开时刻到 11:59 PM
                    prev_dep = stay_time_gen[-1][1] if stay_time_gen else "0000"
                    loc_type_gen.append(1)  # Home -> 1
                    stay_time_gen.append([prev_dep, _to_hhmm("11:59 PM")])
                    a += 1
                else:
                    break

        # 若本文件完全没有收集到任何一段，跳过
        if not stay_time_gen:
            incomplete_generations += int(incomplete_generation)
            continue

        # 细节修饰：首段起点强制 0000，末段终点强制 2400（匹配你的期望）
        stay_time_gen[0][0] = "0000"
        stay_time_gen[-1][1] = "2400"

        # 以本文件为单位合并：构建“一行”的输出
        merged = info.copy()
        stay_times = travel_to_stay(raw_tt) if raw_tt else []
        merged["stay_times"] = stay_times
        merged["id"] = counter
        merged["uuid"] = f_name_no_ext
        merged["loc_type_gen"] = loc_type_gen
        merged["stay_time_gen"] = stay_time_gen
        merged["travel_time_gen"] = stay_to_travel(stay_time_gen)

        data.append(merged)
        counter += 1
        if incomplete_generation:
            incomplete_generations += 1

    print(f"Complete generations: {counter}")
    print(f"Incomplete generations: {incomplete_generations}")
    print(f"Table not found: {table_not_found}")
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="process_outputs")
    parser.add_argument(
        "--type",
        dest="type",
        type=str,
        choices=["survey", "completion"],
        default="survey",
        help="处理类型（当前逻辑仅在 'completion' 分支执行聚合）",
    )
    parser.add_argument(
        "--in_folder",
        dest="in_folder",
        type=str,
        default="outputs/",
        help="输入 JSON 文件夹",
    )
    parser.add_argument(
        "--out_folder",
        dest="out_folder",
        type=str,
        default="outputs_processed/",
        help="输出 CSV 文件夹",
    )
    parser.add_argument(
        "--file_name",
        dest="file_name",
        type=str,
        default="outputs_processed",
        help="输出 CSV 文件名（不含扩展名）",
    )
    parser.add_argument(
        "--fix_missing",
        dest="fix_missing",
        type=bool,
        default=False,
        help="当表格不完整时是否自动补尾（Home -> 1，至 23:59 PM）",
    )

    args = parser.parse_args()

    if args.type == "completion":
        data = process_outputs_completion(
            folder_loc=args.in_folder,
            fix_missing=args.fix_missing,
        )
    else:
        # 与原脚本保持一致：仅当 type == 'completion' 时生成 data
        data = []

    # 输出 CSV
    df = pd.DataFrame(data)
    # 确保输出目录存在（与原逻辑一致性起见，你若不希望自动创建可移除以下两行）
    os.makedirs(args.out_folder, exist_ok=True)
    out_path = f"{args.out_folder}/{args.file_name}.csv"
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
