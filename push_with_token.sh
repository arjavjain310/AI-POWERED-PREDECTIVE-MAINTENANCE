#!/bin/bash
# Push to GitHub using Personal Access Token
# Usage: ./push_with_token.sh YOUR_TOKEN

if [ -z "$1" ]; then
    echo "❌ Please provide your Personal Access Token"
    echo ""
    echo "Usage: ./push_with_token.sh YOUR_TOKEN"
    echo ""
    echo "To get a token:"
    echo "1. Visit: https://github.com/settings/tokens"
    echo "2. Click 'Generate new token (classic)'"
    echo "3. Select scope: 'repo'"
    echo "4. Generate and copy the token"
    echo ""
    exit 1
fi

TOKEN=$1
REPO_URL="https://${TOKEN}@github.com/arjavjain310/AI-POWERED-PREDECTIVE-MAINTENANCE.git"

echo "🚀 Pushing to GitHub..."
git push $REPO_URL main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo "🌐 View at: https://github.com/arjavjain310/AI-POWERED-PREDECTIVE-MAINTENANCE"
else
    echo ""
    echo "❌ Push failed. Please check your token and try again."
fi
