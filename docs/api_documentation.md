API Key Configuration Guide (.env Setup)
This guide explains how to obtain and configure the required API keys for YouTube, Reddit, and Google services. These keys should be stored securely in a .env file and must not be committed to GitHub.

1. YouTube Data API
Purpose: Used to fetch video metadata, comments, transcripts, etc.

Steps to obtain API key:

Go to Google Cloud Console.

Create or select a project.

Navigate to APIs & Services > Library.

Search and enable YouTube Data API v3.

Go to APIs & Services > Credentials.

Click Create Credentials > API key.

.env format:

ini
Copy
Edit
YOUTUBE_API_KEY=your_youtube_api_key
2. Reddit API (via PRAW)
Purpose: Used to collect Reddit posts, comments, and user data.

Steps to obtain credentials:

Visit Reddit App Console.

Click Create App.

Choose script as the app type.

Fill in name, description, and set the redirect URI to http://localhost:8080.

After creation, note down the following:

client_id (displayed under the app name)

client_secret (on the right side)

Your Reddit username and password

.env format:

ini
Copy
Edit
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=SuperBowlAdAnalysis/1.0
REDDIT_USERNAME=your_username
REDDIT_PASSWORD=your_password
3. Google Gemini API
Purpose: Access Google's generative AI models for multimodal analysis (text, image, audio).

Steps to obtain API key:

Go to Google AI Studio.

Log in with your Google account.

Click Create API Key and copy it.

.env format:

ini
Copy
Edit
GOOGLE_API_KEY=your_gemini_api_key
4. Google Cloud Service Account Credentials
Purpose: Used to access other GCP services like Vision API, Speech-to-Text, etc.

Steps to obtain credentials:

Go to Google Cloud Console.

Navigate to IAM & Admin > Service Accounts.

Create a new service account with the required roles.

Go to the Keys tab, and click Add Key > Create new key > JSON.

Save the .json file to your project (e.g., credentials/key.json).

.env format:

pgsql
Copy
Edit
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
In Python:

python
Copy
Edit
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/credentials.json"
Reminder: Always include .env in your .gitignore file to avoid accidental exposure of sensitive keys.

Would you like a downloadable .env.template example to accompany this documentation?
