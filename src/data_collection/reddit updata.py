import pandas as pd
import praw
import time
import re
import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime, timedelta
# 导入 prawcore 异常类，用于更精确地捕获 API 相关的错误
from prawcore.exceptions import PrawcoreException, ResponseException, TooManyRequests


# --- get_super_bowl_ads_from_wikipedia 函数代码（已验证可成功爬取，保持不变） ---
def get_super_bowl_ads_from_wikipedia(start_year=2000, end_year=2025):
    """
    从维基百科页面抓取指定年份范围内的超级碗广告信息。
    通过迭代页面元素流（标题和表格）来确定年份并抓取数据。
    """
    url = "https://en.wikipedia.org/wiki/List_of_Super_Bowl_commercials"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        print(f"DEBUG: 成功获取维基百科页面，状态码: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"DEBUG: 请求维基百科页面失败: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, 'html.parser')
    all_ads_data = []

    all_content_elements = soup.find_all(['h2', 'h3', 'h4', 'table'])
    print(f"DEBUG: 找到 {len(all_content_elements)} 个内容元素（标题和表格）。")

    current_processing_year = None

    for element_idx, element in enumerate(all_content_elements):
        if element.name.startswith('h'):
            header_text = element.get_text(strip=True)
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', header_text)
            if year_match:
                year_from_header = int(year_match.group(0))
                if start_year <= year_from_header <= end_year:
                    current_processing_year = year_from_header
                    print(f"DEBUG: 元素 {element_idx}: 从标题 '{header_text}' 识别到年份: {current_processing_year}")
                else:
                    current_processing_year = None
                    print(
                        f"DEBUG: 元素 {element_idx}: 标题年份 '{header_text}' ({year_from_header}) 不在目标范围 {start_year}-{end_year} 内，跳过此年份段。")
            else:
                print(f"DEBUG: 元素 {element_idx}: 标题 '{header_text}' 未包含有效年份。")

        elif element.name == 'table' and current_processing_year is not None:
            if 'wikitable' not in element.get('class', []):
                continue

            print(f"DEBUG: 元素 {element_idx}: 正在处理年份 {current_processing_year} 下方的表格。")
            rows = element.find_all('tr')
            if not rows or len(rows) < 2:
                print(f"DEBUG: 表格 (Year: {current_processing_year}) 行数不足，跳过。")
                continue

            header_row = rows[0]
            headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
            print(f"DEBUG: 表格 (Year: {current_processing_year}) 的表头: {headers}")

            brand_col_idx = -1
            commercial_col_idx = -1

            possible_brand_headers = ['Brand', 'Company', '品牌', '公司', 'Advertiser/product']
            possible_commercial_headers = ['Commercial', 'Title', 'Ad', '广告', '名称', 'Product/title']

            for idx, header_text in enumerate(headers):
                if brand_col_idx == -1 and any(k.lower() == header_text.lower() for k in possible_brand_headers):
                    brand_col_idx = idx
                if commercial_col_idx == -1 and any(k.lower() == header_text.lower() for k in
                                                    ['title', 'commercial', 'ad', '广告', '名称', 'product/title']):
                    commercial_col_idx = idx

            if brand_col_idx == -1:
                for idx, header_text in enumerate(headers):
                    if header_text.lower() == 'product/title':
                        brand_col_idx = idx
                        break

            print(
                f"DEBUG: 表格 (Year: {current_processing_year}) 识别到的列索引 - Brand: {brand_col_idx}, CommercialName: {commercial_col_idx}")

            if brand_col_idx == -1 or commercial_col_idx == -1:
                print(f"DEBUG: 表格 (Year: {current_processing_year}) 未找到所有关键列（Brand, CommercialName），跳过此表格。")
                continue

            valid_rows_in_table = 0

            current_inherited_year_in_table = current_processing_year

            for row_idx, row in enumerate(rows[1:]):
                final_row_year = current_inherited_year_in_table

                cols = row.find_all(['td', 'th'])

                required_cols_count = max(brand_col_idx, commercial_col_idx) + 1
                if len(cols) < required_cols_count:
                    pass

                row_year_parsed = None

                first_th_in_row = row.find('th')

                if first_th_in_row:
                    year_text_in_cell = first_th_in_row.get_text(strip=True)
                    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', year_text_in_cell)
                    if year_match:
                        row_year_parsed = int(year_match.group(0))
                        final_row_year = row_year_parsed
                        current_inherited_year_in_table = row_year_parsed
                    print(
                        f"DEBUG: 表格 (Element: {element_idx}, Row: {row_idx + 1}) (TH检测): 原始内容: '{year_text_in_cell}', 解析年份: {row_year_parsed}, 当前继承年份: {current_inherited_year_in_table}")
                else:
                    print(f"DEBUG: 表格 (Element: {element_idx}, Row: {row_idx + 1}) (无TH): 未找到TH标签。")

                if final_row_year is None or not (start_year <= final_row_year <= end_year):
                    original_col_texts = [c.get_text(strip=True) for c in cols]
                    print(
                        f"DEBUG: 表格 (Element: {element_idx}, Row: {row_idx + 1}) (原始内容: {original_col_texts}) 因年份 ({final_row_year}) 无效或不在范围 ({start_year}-{end_year}) 被跳过。")
                    continue

                try:
                    brand_raw = cols[brand_col_idx].get_text(strip=True) if brand_col_idx < len(cols) else ""
                    commercial_name_raw = cols[commercial_col_idx].get_text(strip=True) if commercial_col_idx < len(
                        cols) else ""

                    brand = brand_raw
                    commercial_name = commercial_name_raw

                    if brand_col_idx == commercial_col_idx and 'product/title' in [h.lower() for h in headers]:
                        words = commercial_name_raw.split(' ', 1)
                        if ' — ' in commercial_name_raw:
                            parts = commercial_name_raw.split(' — ', 1)
                            brand = parts[0].strip()
                            commercial_name = parts[1].strip()
                        elif ' - ' in commercial_name_raw:
                            parts = commercial_name_raw.split(' - ', 1)
                            brand = parts[0].strip()
                            commercial_name = parts[1].strip()
                        elif len(words) > 1:
                            brand = words[0].strip()
                            commercial_name = words[1].strip()
                        else:
                            brand = commercial_name_raw
                            commercial_name = commercial_name_raw

                    brand = re.sub(r'\[\d+\]', '', brand).strip()
                    commercial_name = re.sub(r'\[\d+\]', '', commercial_name).strip()

                    if not brand or not commercial_name:
                        print(
                            f"DEBUG: 表格 (Element: {element_idx}, Row: {row_idx + 1}) (Year: {final_row_year}) 品牌或广告名称为空，跳过。原始品牌: '{brand_raw}', 原始广告名称: '{commercial_name_raw}', 处理后品牌: '{brand}', 处理后广告名称: '{commercial_name}'")
                        continue

                    all_ads_data.append({
                        'Year': final_row_year,
                        'Brand': brand,
                        'CommercialName': commercial_name
                    })
                    valid_rows_in_table += 1
                except IndexError:
                    print(
                        f"DEBUG: 表格 (Element: {element_idx}, Row: {row_idx + 1}) (Year: {final_row_year}) 出现 IndexError (列索引可能越界)，跳过。原始单元格: {[c.get_text(strip=True) for c in cols]}")
                    pass
                except Exception as e:
                    print(
                        f"DEBUG: 表格 (Element: {element_idx}, Row: {row_idx + 1}) (Year: {final_row_year}) 时发生其他错误: {e}, 行数据: {[c.get_text(strip=True) for c in cols]}")
                    pass
            print(f"DEBUG: 表格 {element_idx} 中成功抓取到 {valid_rows_in_table} 行有效数据。")

    print(f"\nDEBUG: 最终 all_ads_data 列表包含 {len(all_ads_data)} 条数据。")

    if all_ads_data:
        df = pd.DataFrame(all_ads_data)
        df.drop_duplicates(inplace=True)
        df.sort_values(by=['Year', 'Brand', 'CommercialName'], inplace=True)
        print("DEBUG: DataFrame已排序。")
    else:
        df = pd.DataFrame(columns=['Year', 'Brand', 'CommercialName'])
        print("DEBUG: all_ads_data为空，返回一个带空列的DataFrame。")

    return df


# --- Reddit API 配置和相关函数 ---

# 请务必再次检查并确保以下 Reddit API 凭据是正确的！
# 特别是 REDDIT_CLIENT_SECRET，确保它是一个纯净的字符串，不包含制表符 \t
# 例如: REDDIT_CLIENT_SECRET = 'your_actual_secret_without_backslash_t'
# 如果您启用了2FA，请使用“应用专用密码”作为 REDDIT_PASSWORD。
REDDIT_CLIENT_ID = 'ubnBRrAd-8vaup2tWkeoPQ'
REDDIT_CLIENT_SECRET = 'l94SNdHwJzTzZVJwAJvWcXt2kFwuIw'  # <-- 请手动核对并修正此行
REDDIT_USER_AGENT = 'SuperBowlAdCommentScraper/1.0 by Rain'
REDDIT_USERNAME = 'Present-Artichoke-91'
REDDIT_PASSWORD = 'Ly2763225#@!'

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD
)


def clean_search_query(text):
    """清理文本，使其适合作为搜索查询。"""
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def search_reddit_posts_and_comments(brand, commercial_name, year, search_limit=15):
    """
    在Reddit上搜索包含特定超级碗广告名称的帖子，并抓取这些帖子以及其所有评论。
    返回一个字典，其中包含广告信息和所有找到的相关帖子及其评论。

    参数:
        brand (str): 广告品牌名。
        commercial_name (str): 广告名称。
        year (int): 广告年份。
        search_limit (int): 每个查询最多返回的帖子数量。PRAW会自动处理分页。
    """
    query_variants = [
        f'"{clean_search_query(commercial_name)}" "{clean_search_query(brand)}" Super Bowl {year} ad',
        f'"{clean_search_query(brand)}" "{clean_search_query(commercial_name)}" Super Bowl ad',
        f'"{clean_search_query(commercial_name)}" Super Bowl {year}',
        f'"{clean_search_query(brand)}" Super Bowl {year} ad'
    ]

    ad_reddit_data = {
        "AdYear": year,
        "AdBrand": brand,
        "AdCommercialName": commercial_name,
        "RedditDiscussions": []
    }

    print(f"\n正在搜索 Reddit 上关于广告 '{brand} - {commercial_name}' ({year}) 的帖子和评论...")

    target_subreddits = reddit.subreddit('SuperBowl+commercials+advertising+television+marketing')

    total_submissions_found_for_ad = 0

    # 引入重试机制，对每个广告的搜索尝试多次
    max_ad_retries = 3
    current_ad_retry = 0

    while current_ad_retry < max_ad_retries:
        try:
            for query_idx, query in enumerate(query_variants):
                print(
                    f"  [{query_idx + 1}/{len(query_variants)}] 尝试搜索查询: {query} (广告重试 {current_ad_retry + 1}/{max_ad_retries})")

                submissions_for_query_count = 0

                # PRAW的search()方法本身就会返回与查询相关的帖子
                submissions_generator = target_subreddits.search(query, limit=search_limit)

                for submission in submissions_generator:
                    # 避免重复添加已经处理过的帖子
                    if any(s['SubmissionID'] == submission.id for s in ad_reddit_data["RedditDiscussions"]):
                        continue

                    if not submission.title or not submission.url:
                        continue

                    print(f"    找到相关 Reddit 帖子: '{submission.title}' (ID: {submission.id})")
                    print(f"    帖子链接: {submission.url}")

                    submission_data = {
                        'SubmissionID': submission.id,
                        'SubmissionTitle': submission.title,
                        'SubmissionURL': submission.url,
                        'SubmissionAuthor': str(submission.author) if submission.author else '[deleted]',
                        'SubmissionScore': submission.score,
                        'SubmissionNumComments': submission.num_comments,
                        'SubmissionCreatedUTC': pd.to_datetime(submission.created_utc, unit='s').isoformat(),
                        'SubmissionText': submission.selftext if submission.selftext else '[No selftext]',
                        'SubmissionPermalink': submission.permalink,
                        'Comments': []
                    }

                    # 对评论获取也引入重试机制
                    max_comment_retries = 2
                    current_comment_retry = 0
                    comments_fetched_successfully = False

                    while current_comment_retry < max_comment_retries:
                        try:
                            # 确保替换“更多评论”链接
                            submission.comments.replace_more(limit=None)
                            comments_fetched_successfully = True
                            break  # 成功获取评论，跳出评论重试循环
                        except TooManyRequests as tmr_e:
                            current_comment_retry += 1
                            print(
                                f"    获取评论时遇到速率限制 (TooManyRequests): {tmr_e}. 暂停 {10 * current_comment_retry} 秒并重试评论获取...")
                            time.sleep(10 * current_comment_retry)
                        except PrawcoreException as e:
                            print(f"    获取评论时发生 Prawcore 错误: {e}. 跳过此帖子的评论。")
                            break  # 遇到其他Prawcore错误，跳过当前评论的获取
                        except Exception as e:
                            print(f"    获取评论时发生其他错误: {e}. 跳过此帖子的评论。")
                            break  # 遇到其他错误，跳过当前评论的获取

                    if not comments_fetched_successfully:
                        print(f"    未能成功获取帖子 '{submission.title}' (ID: {submission.id}) 的评论。跳过此帖子。")
                        continue  # 跳过当前帖子，处理下一个 submission

                    comments_in_submission_count = 0
                    for comment in submission.comments.list():
                        if isinstance(comment, praw.models.Comment) and comment.body and \
                                len(comment.body) > 10 and str(comment.author) != 'AutoModerator' and \
                                comment.body != '[deleted]' and comment.body != '[removed]':
                            submission_data['Comments'].append({
                                'CommentID': comment.id,
                                'CommentParentID': comment.parent_id if comment.parent_id else 'None',
                                'CommentAuthor': str(comment.author) if comment.author else '[deleted]',
                                'CommentScore': comment.score,
                                'CommentText': comment.body,
                                'CommentCreatedUTC': pd.to_datetime(comment.created_utc, unit='s').isoformat(),
                                'CommentPermalink': comment.permalink
                            })
                            comments_in_submission_count += 1

                    ad_reddit_data["RedditDiscussions"].append(submission_data)
                    print(f"    帖子 '{submission.title}' 抓取到 {comments_in_submission_count} 条评论。")
                    submissions_for_query_count += 1
                    total_submissions_found_for_ad += 1

                    time.sleep(5)  # 每次处理一个帖子后暂停
                print(f"  查询 '{query}' 实际处理了 {submissions_for_query_count} 个帖子。")
                time.sleep(10)  # 每个查询变体处理完毕后暂停

            # 如果成功完成所有查询变体，则跳出重试循环
            return ad_reddit_data

        except TooManyRequests as tmr_e:
            current_ad_retry += 1
            print(
                f"搜索广告 '{brand} - {commercial_name}' 时遇到速率限制 (TooManyRequests): {tmr_e}. 第 {current_ad_retry} 次广告重试...")
            if current_ad_retry < max_ad_retries:
                time.sleep(30 * current_ad_retry)  # 增加退避时间，防止再次立即触发限制
            else:
                print(f"已达到最大广告重试次数 ({max_ad_retries})，跳过广告 '{brand} - {commercial_name}'。")
                return {
                    "AdYear": year,
                    "AdBrand": brand,
                    "AdCommercialName": commercial_name,
                    "Error": "TooManyRequests_MaxRetriesReached"
                }
        except ResponseException as resp_e:
            print(f"搜索广告 '{brand} - {commercial_name}' 时发生 Reddit API 响应错误: {resp_e}")
            if resp_e.response.status_code == 401 or resp_e.response.status_code == 403:
                print(
                    "可能是 Reddit 认证失败。请检查您的 REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD 是否正确。")
                print("如果开启了2FA，请务必使用“应用专用密码”而非主密码。")
                return {
                    "AdYear": year,
                    "AdBrand": brand,
                    "AdCommercialName": commercial_name,
                    "Error": "Reddit Authentication Failed"
                }
            else:
                current_ad_retry += 1
                print(f"其他 Reddit API 响应错误。第 {current_ad_retry} 次广告重试...")
                if current_ad_retry < max_ad_retries:
                    time.sleep(15 * current_ad_retry)
                else:
                    print(f"已达到最大广告重试次数 ({max_ad_retries})，跳过广告 '{brand} - {commercial_name}'。")
                    return {
                        "AdYear": year,
                        "AdBrand": brand,
                        "AdCommercialName": commercial_name,
                        "Error": "ResponseException_MaxRetriesReached"
                    }
        except PrawcoreException as core_e:
            current_ad_retry += 1
            print(f"搜索广告 '{brand} - {commercial_name}' 时发生 Reddit Prawcore 错误: {core_e}. 第 {current_ad_retry} 次广告重试...")
            if current_ad_retry < max_ad_retries:
                time.sleep(15 * current_ad_retry)
            else:
                print(f"已达到最大广告重试次数 ({max_ad_retries})，跳过广告 '{brand} - {commercial_name}'。")
                return {
                    "AdYear": year,
                    "AdBrand": brand,
                    "AdCommercialName": commercial_name,
                    "Error": "PrawcoreException_MaxRetriesReached"
                }
        except Exception as e:
            current_ad_retry += 1
            print(f"搜索广告 '{brand} - {commercial_name}' 时发生其他错误: {e}. 第 {current_ad_retry} 次广告重试...")
            if current_ad_retry < max_ad_retries:
                time.sleep(10 * current_ad_retry)
            else:
                print(f"已达到最大广告重试次数 ({max_ad_retries})，跳过广告 '{brand} - {commercial_name}'。")
                return {
                    "AdYear": year,
                    "AdBrand": brand,
                    "AdCommercialName": commercial_name,
                    "Error": "OtherError_MaxRetriesReached"
                }

    print(f"广告 '{brand} - {commercial_name}' ({year}) 总共找到并处理了 {total_submissions_found_for_ad} 个相关帖子。")
    return ad_reddit_data


def main_reddit_scraper():
    print("当前工作目录:", os.getcwd())  # 打印当前工作目录，方便确认输出文件位置

    # 步骤1：从维基百科获取广告列表
    print("第一步：从维基百科爬取超级碗广告列表...")
    ads_df = get_super_bowl_ads_from_wikipedia(start_year=2000, end_year=2025)

    if ads_df.empty:
        print("未能从维基百科获取到广告列表。请检查网络或维基百科页面结构。")
        return

    print(f"成功从维基百科获取到 {len(ads_df)} 条广告数据。")
    print(ads_df.head())

    output_dir = "reddit_ad_data_json"  # 定义输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    else:
        print(f"输出目录已存在: {output_dir}")  # 明确目录已存在

    progress_file = "progress.txt"
    processed_ad_indices = set()

    # 在加载进度文件时，确保文件存在且可读
    try:
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                for line in f:
                    try:
                        idx = int(line.strip())
                        if 0 <= idx < len(ads_df):  # 确保索引在有效范围内
                            processed_ad_indices.add(idx)
                    except ValueError:
                        print(f"警告：进度文件中包含无效行 '{line.strip()}'，已跳过。")
                        continue
            print(f"从 '{progress_file}' 加载了 {len(processed_ad_indices)} 条已处理广告记录。")
        else:
            print(f"未找到进度文件 '{progress_file}'，将从头开始。")
    except Exception as e:
        print(f"加载进度文件时发生错误: {e}")
        processed_ad_indices = set()  # 出现错误则清空已处理列表，从头开始

    print("\n第二步：开始在Reddit上搜索并爬取帖子和评论，并保存为JSON文件...")
    total_ads = len(ads_df)

    for index, row in ads_df.iterrows():
        # 如果当前广告索引在 processed_ad_indices 中，跳过
        if index in processed_ad_indices:
            print(f"--- 跳过已处理广告 {index + 1}/{total_ads}: '{row['CommercialName']}' ({row['Year']}) ---")
            continue

        year = row['Year']
        brand = row['Brand']
        commercial_name = row['CommercialName']

        if not brand or not commercial_name:
            print(f"跳过无效广告条目：Year={year}, Brand={brand}, CommercialName={commercial_name} (品牌或广告名称为空)")
            # 记录此条目为已处理，避免下次重复跳过
            with open(progress_file, 'a') as f:
                f.write(f"{index}\n")
            continue

        print(f"\n--- 正在处理广告 {index + 1}/{total_ads}: '{commercial_name}' ({year}) ---")

        # 对文件名进行更严格的清理，确保合法性
        clean_commercial_name = re.sub(r'[^\w\s-]', '', commercial_name)  # 移除所有非字母数字、空格和破折号的字符
        clean_commercial_name = clean_commercial_name.replace(' ', '_')
        clean_commercial_name = re.sub(r'_{2,}', '_', clean_commercial_name)  # 将多个下划线替换成一个
        clean_commercial_name = clean_commercial_name.strip('_')

        # 确保文件名不会太长，否则可能导致文件系统错误
        if len(clean_commercial_name) > 100:
            clean_commercial_name = clean_commercial_name[:100]

        file_name = f"{year}_{clean_commercial_name}.json"
        output_file_path = os.path.join(output_dir, file_name)

        # 检查文件是否已经存在，如果存在则跳过
        if os.path.exists(output_file_path):
            print(f"文件 '{output_file_path}' 已存在，跳过此广告的数据抓取和保存。")
            with open(progress_file, 'a') as f:
                f.write(f"{index}\n")
            continue

        ad_data_for_json = search_reddit_posts_and_comments(brand, commercial_name, year, search_limit=15)

        # 无论 search_reddit_posts_and_comments 返回什么，都尝试将其保存到文件。
        # 只有在遇到 Reddit 认证失败时才停止整个脚本。
        if "Error" in ad_data_for_json and ad_data_for_json["Error"] == "Reddit Authentication Failed":
            print(f"由于Reddit认证错误，停止处理。")
            return  # 认证失败是致命错误，应停止

        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(ad_data_for_json, f, ensure_ascii=False, indent=4)

            # 根据 ad_data_for_json 的内容，提供更详细的保存成功信息
            if "Error" in ad_data_for_json:
                print(
                    f"广告 '{commercial_name}' ({year}) 处理完成，保存文件到 '{output_file_path}' (包含错误信息: {ad_data_for_json['Error']})")
            elif "RedditDiscussions" in ad_data_for_json and ad_data_for_json["RedditDiscussions"]:
                print(f"成功将广告 '{commercial_name}' ({year}) 的数据保存到 '{output_file_path}'")
            else:  # 文件没有错误，但也没有 RedditDiscussions (可能意味着没有找到相关讨论)
                print(f"广告 '{commercial_name}' ({year}) 处理完成，保存文件到 '{output_file_path}' (未找到Reddit讨论或为空)")

            # 只有成功保存到文件后，才将索引写入进度文件
            with open(progress_file, 'a') as f:
                f.write(f"{index}\n")
            print(f"进度已保存：广告索引 {index} 已处理。")

        except Exception as e:
            print(f"保存文件 '{output_file_path}' 时发生错误: {e}")
            # 如果保存失败，不写入进度文件，以便下次重试

        time.sleep(10)  # 每次处理完一个广告（无论是否找到讨论或遇到错误并保存）后暂停，以降低整体请求频率

    print(f"\n所有广告处理完毕。")


if __name__ == "__main__":
    main_reddit_scraper()