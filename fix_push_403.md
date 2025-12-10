# Fixing 403 Permission Denied Error

## Common Causes & Solutions

### Issue: 403 Permission Denied
This means your authentication token doesn't have the right permissions or is invalid.

## Solution 1: Create a New Token with Correct Permissions

1. **Go to GitHub Settings:**
   https://github.com/settings/tokens

2. **Generate New Token:**
   - Click "Generate new token" → "Generate new token (classic)"
   - Name: "Predictive Maintenance Push"
   - Expiration: Choose 90 days or custom
   - **IMPORTANT: Select these scopes:**
     - ✅ `repo` (Full control of private repositories)
       - This includes: `repo:status`, `repo_deployment`, `public_repo`, `repo:invite`, `security_events`
   - Click "Generate token"
   - **COPY THE TOKEN IMMEDIATELY** (you won't see it again!)

3. **Use the Token:**
   ```bash
   cd "/Users/arjavjain/Desktop/MAJOR PROJECT"
   git push -u origin main
   ```
   - Username: `arjavjain310`
   - Password: **Paste your new token**

## Solution 2: Use Token in URL (Alternative)

If interactive prompt doesn't work:

```bash
cd "/Users/arjavjain/Desktop/MAJOR PROJECT"
git remote set-url origin https://YOUR_TOKEN@github.com/arjavjain310/AI-POWERED-PREDECTIVE-MAINTENANCE.git
git push -u origin main
```

Replace `YOUR_TOKEN` with your actual token.

## Solution 3: Check Repository Access

1. Make sure you're logged into GitHub as `arjavjain310`
2. Verify the repository exists: https://github.com/arjavjain310/AI-POWERED-PREDECTIVE-MAINTENANCE
3. Check if repository is private (if so, token needs `repo` scope)

## Solution 4: Use SSH Instead

If HTTPS keeps failing, switch to SSH:

```bash
cd "/Users/arjavjain/Desktop/MAJOR PROJECT"
git remote set-url origin git@github.com:arjavjain310/AI-POWERED-PREDECTIVE-MAINTENANCE.git
git push -u origin main
```

(Requires SSH keys to be set up)

## Quick Fix Script

Run this after getting your new token:

```bash
cd "/Users/arjavjain/Desktop/MAJOR PROJECT"
read -sp "Enter your GitHub token: " TOKEN
echo ""
git remote set-url origin https://${TOKEN}@github.com/arjavjain310/AI-POWERED-PREDECTIVE-MAINTENANCE.git
git push -u origin main
```

## Most Common Issue

**The token must have `repo` scope enabled!** Without it, you'll get 403 errors.

Make sure when creating the token, you check the `repo` checkbox.
