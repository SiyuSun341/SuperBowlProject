import praw
import pandas as pd
from datetime import datetime
import logging
import os

# 设置保存路径
SAVE_DIR = r"S:\Documents\2025 - Purdue\AAAAAPURDUE\5 Capstone\SuperBowlProject\tests\Raddit\test_data_collection"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(SAVE_DIR, 'reddit_collection.log')),
        logging.StreamHandler()
    ]
)

def setup_reddit_client():
    """设置Reddit API客户端"""
    try:
        reddit = praw.Reddit(
<<<<<<< HEAD
            client_id="",
            client_secret="",
            user_agent="script:SuperBowl:v1.0 (by /u/your_reddit_username)"
=======
            client_id="your client id",
            client_secret="your secret",
            user_agent=" agent"
>>>>>>> 72087d38e23fdea35548f560e950e62eab3d0d58
        )
        return reddit
    except Exception as e:
        logging.error(f"Failed to setup Reddit client: {str(e)}")
        raise

def collect_comments(post, max_comments=100):
    """
    收集帖子的评论
    
    Args:
        post: Reddit帖子对象
        max_comments: 最大评论数量
    
    Returns:
        list: 评论数据列表
    """
    comments_data = []
    try:
        post.comments.replace_more(limit=0)  # 只获取顶层评论
        for comment in post.comments.list()[:max_comments]:
            comment_data = {
                'post_id': post.id,
                'comment_id': comment.id,
                'author': str(comment.author),
                'body': comment.body,
                'score': comment.score,
                'created_utc': datetime.fromtimestamp(comment.created_utc),
                'permalink': comment.permalink
            }
            comments_data.append(comment_data)
    except Exception as e:
        logging.error(f"Error collecting comments for post {post.id}: {str(e)}")
    return comments_data

def collect_superbowl_posts(reddit, keywords, limit=10):
    """
    收集包含特定关键词的Reddit帖子和评论
    
    Args:
        reddit: Reddit API客户端实例
        keywords: 关键词列表
        limit: 每个关键词收集的帖子数量
    
    Returns:
        tuple: (帖子DataFrame, 评论DataFrame)
    """
    posts_data = []
    comments_data = []
    
    try:
        # 构建搜索查询
        search_query = " OR ".join([f'"{keyword}"' for keyword in keywords])
        
        # 在多个相关subreddit中搜索
        subreddits = ['SuperBowl', 'nfl', 'advertising', 'marketing']
        
        for subreddit_name in subreddits:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                search_results = subreddit.search(search_query, limit=limit, sort='relevance', time_filter='year')
                
                for post in search_results:
                    # 收集帖子数据
                    post_data = {
                        'id': post.id,
                        'title': post.title,
                        'author': str(post.author),
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'url': post.url,
                        'subreddit': subreddit_name,
                        'selftext': post.selftext,
                        'permalink': post.permalink
                    }
                    posts_data.append(post_data)
                    
                    # 收集评论数据
                    post_comments = collect_comments(post)
                    comments_data.extend(post_comments)
                    
                    logging.info(f"Collected post: {post.title} with {len(post_comments)} comments")
                    
            except Exception as e:
                logging.error(f"Error collecting from r/{subreddit_name}: {str(e)}")
                continue
        
        # 转换为DataFrame
        posts_df = pd.DataFrame(posts_data)
        comments_df = pd.DataFrame(comments_data)
        
        # 确保保存目录存在
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        # 保存到CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        posts_file = os.path.join(SAVE_DIR, f'superbowl_posts_{timestamp}.csv')
        comments_file = os.path.join(SAVE_DIR, f'superbowl_comments_{timestamp}.csv')
        
        posts_df.to_csv(posts_file, index=False)
        comments_df.to_csv(comments_file, index=False)
        
        logging.info(f"Successfully saved {len(posts_df)} posts to {posts_file}")
        logging.info(f"Successfully saved {len(comments_df)} comments to {comments_file}")
        
        return posts_df, comments_df
        
    except Exception as e:
        logging.error(f"Error in collect_superbowl_posts: {str(e)}")
        raise

def main():
    """主函数"""
    try:
        # 设置Reddit客户端
        reddit = setup_reddit_client()
        
        # 定义搜索关键词
        keywords = [
            "Super Bowl Ad",
            "Rogue Ridge",
            "Super Bowl Commercial",
            "Super Bowl 2024 Ad",
            "Super Bowl Marketing"
        ]
        
        # 收集帖子和评论
        posts_df, comments_df = collect_superbowl_posts(reddit, keywords, limit=10)
        
        # 打印收集结果
        print(f"\nCollected {len(posts_df)} posts and {len(comments_df)} comments")
        print("\nSample of collected posts:")
        print(posts_df[['title', 'subreddit', 'score', 'num_comments']].head())
        print("\nSample of collected comments:")
        print(comments_df[['post_id', 'author', 'score']].head())
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
