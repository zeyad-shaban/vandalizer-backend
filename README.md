---
title: Vandalizer Backend
emoji: 🖌️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

sudo systemctl stop redis-server
# Commands
1. Redis
> sudo systemctl stop redis-server
> redis-server --bind 0.0.0.0 --protected-mode no

2. celery worker
> watchfiles --filter python "cmd /c celery -A tasks worker -P solo" ./tasks.py

3. Run main.py
> uvicorn main:app --reload