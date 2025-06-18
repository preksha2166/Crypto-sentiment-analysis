"""
Data collection module for gathering cryptocurrency-related data from various sources.
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any

import tweepy
import praw
from telegram import Bot
import discord
from github import Github

from ..config import API_KEYS, COLLECTION_SETTINGS

logger = logging.getLogger(__name__)

class BaseCollector(ABC):
    """Base class for all data collectors."""
    
    def __init__(self):
        self.last_collection_time = None
        self.rate_limit_pause = 60  # Default rate limit pause in seconds
    
    @abstractmethod
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect data from the source."""
        pass
    
    @abstractmethod
    async def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process collected data into a standardized format."""
        pass

class TwitterCollector(BaseCollector):
    """Collector for Twitter data."""
    
    def __init__(self):
        super().__init__()
        self.api = tweepy.Client(
            bearer_token=API_KEYS["twitter"],
            wait_on_rate_limit=True
        )
        self.settings = COLLECTION_SETTINGS["twitter"]
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect tweets based on configured keywords."""
        try:
            tweets = []
            for keyword in self.settings["track_keywords"]:
                response = self.api.search_recent_tweets(
                    query=keyword,
                    max_results=self.settings["max_tweets_per_request"],
                    tweet_fields=["created_at", "public_metrics", "author_id"],
                    user_fields=["username", "public_metrics"],
                    expansions=["author_id"]
                )
                
                if response.data:
                    tweets.extend(response.data)
            
            return await self.process(tweets)
        except Exception as e:
            logger.error(f"Error collecting Twitter data: {str(e)}")
            return []
    
    async def process(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process tweets into standardized format."""
        processed_tweets = []
        for tweet in tweets:
            processed_tweets.append({
                "source": "twitter",
                "id": tweet.id,
                "text": tweet.text,
                "created_at": tweet.created_at,
                "metrics": tweet.public_metrics,
                "author_id": tweet.author_id,
                "collected_at": datetime.utcnow()
            })
        return processed_tweets

class RedditCollector(BaseCollector):
    """Collector for Reddit data."""
    
    def __init__(self):
        super().__init__()
        self.reddit = praw.Reddit(
            client_id=API_KEYS["reddit"]["client_id"],
            client_secret=API_KEYS["reddit"]["client_secret"],
            user_agent=API_KEYS["reddit"]["user_agent"]
        )
        self.settings = COLLECTION_SETTINGS["reddit"]
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect posts from configured subreddits."""
        try:
            posts = []
            for subreddit_name in self.settings["subreddits"]:
                subreddit = self.reddit.subreddit(subreddit_name)
                for post in subreddit.new(limit=self.settings["max_posts_per_request"]):
                    posts.append(post)
            
            return await self.process(posts)
        except Exception as e:
            logger.error(f"Error collecting Reddit data: {str(e)}")
            return []
    
    async def process(self, posts: List[Any]) -> List[Dict[str, Any]]:
        """Process Reddit posts into standardized format."""
        processed_posts = []
        for post in posts:
            processed_posts.append({
                "source": "reddit",
                "id": post.id,
                "title": post.title,
                "text": post.selftext,
                "created_at": datetime.fromtimestamp(post.created_utc),
                "metrics": {
                    "score": post.score,
                    "upvote_ratio": post.upvote_ratio,
                    "num_comments": post.num_comments
                },
                "subreddit": post.subreddit.display_name,
                "author": post.author.name if post.author else None,
                "collected_at": datetime.utcnow()
            })
        return processed_posts

class TelegramCollector(BaseCollector):
    """Collector for Telegram data."""
    
    def __init__(self):
        super().__init__()
        self.bot = Bot(token=API_KEYS["telegram"])
        self.settings = COLLECTION_SETTINGS["telegram"]
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect messages from configured channels."""
        try:
            messages = []
            for channel in self.settings["channels"]:
                # Implementation depends on Telegram API access
                # This is a placeholder for the actual implementation
                pass
            return await self.process(messages)
        except Exception as e:
            logger.error(f"Error collecting Telegram data: {str(e)}")
            return []
    
    async def process(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process Telegram messages into standardized format."""
        processed_messages = []
        for message in messages:
            processed_messages.append({
                "source": "telegram",
                "id": message.get("message_id"),
                "text": message.get("text"),
                "created_at": message.get("date"),
                "channel": message.get("chat", {}).get("title"),
                "author": message.get("from", {}).get("username"),
                "collected_at": datetime.utcnow()
            })
        return processed_messages

class GitHubCollector(BaseCollector):
    """Collector for GitHub data."""
    
    def __init__(self):
        super().__init__()
        self.github = Github(API_KEYS["github"])
        self.settings = COLLECTION_SETTINGS["github"]
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect commits from configured repositories."""
        try:
            commits = []
            for repo_name in self.settings["repositories"]:
                repo = self.github.get_repo(repo_name)
                for commit in repo.get_commits()[:self.settings["max_commits_per_request"]]:
                    commits.append(commit)
            
            return await self.process(commits)
        except Exception as e:
            logger.error(f"Error collecting GitHub data: {str(e)}")
            return []
    
    async def process(self, commits: List[Any]) -> List[Dict[str, Any]]:
        """Process GitHub commits into standardized format."""
        processed_commits = []
        for commit in commits:
            processed_commits.append({
                "source": "github",
                "id": commit.sha,
                "message": commit.commit.message,
                "created_at": commit.commit.author.date,
                "repository": commit.repository.full_name,
                "author": commit.author.login if commit.author else None,
                "metrics": {
                    "additions": commit.stats.additions,
                    "deletions": commit.stats.deletions,
                    "total": commit.stats.total
                },
                "collected_at": datetime.utcnow()
            })
        return processed_commits 