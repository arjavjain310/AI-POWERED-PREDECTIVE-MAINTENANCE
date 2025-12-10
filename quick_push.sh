#!/bin/bash
# Quick push script - fixes 403 error
# Usage: ./quick_push.sh

echo "🔧 Fixing 403 Permission Error"
echo "================================"
echo ""

cd "$(dirname "$0")"

echo "Step 1: Getting your Personal Access Token"
echo "-------------------------------------------"
echo "1. Visit: https://github.com/settings/tokens"
echo "2. Click 'Generate new token (classic)'"
echo "3. Name: 'Predictive Maintenance'"
echo "4. ✅ CHECK 'repo' scope (IMPORTANT!)"
echo "5. Generate and copy the token"
echo ""
read -sp "Paste your token here: " TOKEN
echo ""
echo ""

if [ -z "$TOKEN" ]; then
    echo "❌ No token provided. Exiting."
    exit 1
fi

echo "Step 2: Updating remote URL with token..."
git remote set-url origin https://${TOKEN}@github.com/arjavjain310/AI-POWERED-PREDECTIVE-MAINTENANCE.git

echo "Step 3: Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS! Project pushed to GitHub!"
    echo "🌐 View at: https://github.com/arjavjain310/AI-POWERED-PREDECTIVE-MAINTENANCE"
    echo ""
    # Remove token from remote URL for security
    git remote set-url origin https://github.com/arjavjain310/AI-POWERED-PREDECTIVE-MAINTENANCE.git
    echo "🔒 Removed token from git config for security"
else
    echo ""
    echo "❌ Push failed. Common issues:"
    echo "   - Token doesn't have 'repo' scope"
    echo "   - Token is expired"
    echo "   - Repository doesn't exist or you don't have access"
    echo ""
    echo "Try creating a new token with 'repo' scope enabled."
fi
