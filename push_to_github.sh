#!/bin/bash
# Script to push project to GitHub
# Usage: ./push_to_github.sh

echo "🚀 Pushing AI-Powered Predictive Maintenance Project to GitHub"
echo "================================================================"
echo ""

cd "$(dirname "$0")"

# Check if remote is configured
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "❌ Remote not configured. Setting up..."
    git remote add origin https://github.com/arjavjain310/AI-POWERED-PREDECTIVE-MAINTENANCE.git
fi

echo "📋 Repository: https://github.com/arjavjain310/AI-POWERED-PREDECTIVE-MAINTENANCE.git"
echo ""
echo "Current status:"
git status --short | head -10
echo ""

# Check if there are uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  You have uncommitted changes. Committing them..."
    git add -A
    git commit -m "Update: Karnataka wind farms predictive maintenance system"
fi

echo ""
echo "📤 Pushing to GitHub..."
echo ""
echo "You will be prompted for credentials:"
echo "  Username: arjavjain310"
echo "  Password: Use a Personal Access Token (not your GitHub password)"
echo ""
echo "To create a token: https://github.com/settings/tokens"
echo ""

git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo "🌐 View at: https://github.com/arjavjain310/AI-POWERED-PREDECTIVE-MAINTENANCE"
else
    echo ""
    echo "❌ Push failed. This usually means:"
    echo "   1. Authentication required - use Personal Access Token"
    echo "   2. Repository permissions issue"
    echo ""
    echo "Alternative: Use GitHub Desktop or web interface"
fi

