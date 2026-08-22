---
name: cega-interaction
description: Standardized workflows for interacting with the CEGA application running at localhost:8000
version: 1.0.0
author: Dan
---

# CEGA Interaction Skill

## Description
This skill provides standardized workflows for interacting with the CEGA (Crystallised Evolutionary Gra... application running at localhost:8000. It includes setup instructions, common operations, and troubleshooting guidance.

## Setup Instructions
1. Ensure the CEGA application is running at localhost:8000
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables:
   ```bash
   export CEGA_HOST=localhost
   export CEGA_PORT=8000
   ```

## Common Operations

### Check Application Status
```bash
curl http://localhost:8000/health
```

### View Available Endpoints
```bash
curl http://localhost:8000/api
```

### Run Evolutionary Algorithm
```bash
curl -X POST http://localhost:8000/api/run \
  -H "Content-Type: application/json" \
  -d '{"population_size": 100, "generations": 50}'
```

### Monitor Progress
```bash
curl http://localhost:8000/api/progress
```

## Troubleshooting Guide

### Application Not Running
- Verify the CEGA application is started:
  ```bash
  ps aux | grep cega
  ```
- Start the application if not running:
  ```bash
  python manage.py runserver 8000
  ```

### Connection Refused
- Check if port 8000 is occupied:
  ```bash
  lsof -i :8000
  ```
- Kill process using port 8000 if necessary:
  ```bash
  kill -9 $(lsof -t -i :8000)
  ```

### API Errors
- Review application logs:
  ```bash
  tail -f logs/cega.log
  ```
- Check for syntax errors in Python files:
  ```bash
  python -m py_compile *.py
  ```

## References
- CEGA Documentation: https://github.com/your-org/CEGA/wiki
- API Specification: https://github.com/your-org/CEGA/wiki/API-Reference

## Scripts
None available.