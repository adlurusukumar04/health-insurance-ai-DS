# GitHub Setup Guide — Health Insurance AI Platform

## Step 1: Initialize Local Git Repository

```bash
cd health-insurance-ai
git init
git add .
git commit -m "feat: initial project structure — health insurance AI platform"
```

## Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `health-insurance-ai`
3. Description: `AI/ML platform for health insurance claim processing, fraud detection & personalization`
4. Visibility: Public (for portfolio)
5. Do NOT initialize with README (we have one)
6. Click **Create repository**

## Step 3: Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/health-insurance-ai.git
git branch -M main
git push -u origin main
```

## Step 4: Add GitHub Secrets (for CI/CD)

Go to: Settings → Secrets and variables → Actions → New repository secret

| Secret Name           | Value                          |
|-----------------------|--------------------------------|
| AWS_ACCESS_KEY_ID     | Your AWS access key            |
| AWS_SECRET_ACCESS_KEY | Your AWS secret key            |
| AWS_ACCOUNT_ID        | Your 12-digit AWS account ID   |

## Step 5: Set Up Branch Protection

Settings → Branches → Add rule:
- Branch name: `main`
- ✅ Require pull request reviews before merging
- ✅ Require status checks to pass (lint, test, model-smoke)
- ✅ Require branches to be up to date

## Step 6: Add GitHub Topics (for discoverability)

Settings → Topics → Add:
`machine-learning`, `health-insurance`, `fraud-detection`, `nlp`, `xgboost`,
`tensorflow`, `fastapi`, `aws`, `airflow`, `python`

## Step 7: Create Project Branches

```bash
git checkout -b develop
git push -u origin develop

git checkout -b feature/fraud-detection
git push -u origin feature/fraud-detection
```

## Recommended Repository Settings

- Enable GitHub Pages for README preview
- Add `.gitignore` for Python (already included)
- Pin repository to your GitHub profile

## Portfolio Tips

1. Add a demo GIF/screenshot to README (Power BI dashboard screenshot)
2. Link to notebook viewer: https://nbviewer.org/github/YOUR_USERNAME/health-insurance-ai
3. Add to LinkedIn profile under "Projects"
4. Write a Medium/Dev.to article about your architecture choices
