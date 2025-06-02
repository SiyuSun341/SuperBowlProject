import praw
import pandas as pd
from datetime import datetime
import logging
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/reddit_collection.log'),
        logging.StreamHandler()
    ]
)

def setup_reddit_client():
    """设置Reddit API客户端"""
    try:
        reddit = praw.Reddit(
            client_id="your_client_id",
            client_secret="your_client_secret",
            user_agent="your_ user_agent"
        )
        return reddit
    except Exception as e:
        logging.error(f"Failed to setup Reddit client: {str(e)}")
        raise

def collect_superbowl_posts(reddit, keywords, limit=10):
    """
    收集包含特定关键词的Reddit帖子
    
    Args:
        reddit: Reddit API客户端实例
        keywords: 关键词列表
        limit: 每个关键词收集的帖子数量
    
    Returns:
        DataFrame: 包含帖子信息的DataFrame
    """
    posts_data = []
    
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
                    
            except Exception as e:
                logging.error(f"Error collecting posts from r/{subreddit_name}: {str(e)}")
                continue
        
        # 转换为DataFrame
        df = pd.DataFrame(posts_data)
        
        # 保存到CSV
        output_dir = 'data/raw/reddit/posts'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'superbowl_posts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        df.to_csv(output_file, index=False)
        
        logging.info(f"Successfully collected {len(df)} posts and saved to {output_file}")
        return df
        
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
        
        # 收集帖子
        posts_df = collect_superbowl_posts(reddit, keywords, limit=10)
        
        # 打印收集到的帖子数量
        print(f"\nCollected {len(posts_df)} posts")
        print("\nSample of collected posts:")
        print(posts_df[['title', 'subreddit', 'score', 'num_comments']].head())
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
