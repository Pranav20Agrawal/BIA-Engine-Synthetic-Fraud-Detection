# test_auth.py

import praw
import sys

print("--- Reddit Authentication Test ---")
print("Attempting to connect using the [BIA_SCRAPER] section of your praw.ini file...")

try:
    # This line reads your praw.ini and attempts to create a connection
    reddit = praw.Reddit("BIA_SCRAPER")
    
    # This line tries to access your user profile, which requires a successful login
    user = reddit.user.me()
    
    # If the lines above work without error, your credentials are correct
    print("\n✅ SUCCESS: Authentication successful!")
    print(f"Logged in as user: {user}")
    print("\nYour credentials in praw.ini are correct. You can now run the main scraper.")

except Exception as e:
    # If any part of the authentication fails, we will end up here
    print("\n❌ FAILED: Authentication failed.")
    print("This confirms there is an issue with the credentials in your praw.ini file.")
    print("\nDetailed Error:")
    print(e)
    if "401" in str(e):
        print("\nHINT: A '401' error means 'Unauthorized'. Please carefully re-check every character in your praw.ini file.")
    sys.exit(1)