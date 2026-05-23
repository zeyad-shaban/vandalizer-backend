sudo systemctl stop redis-server

watchfiles --filter python "celery -A tasks worker -P solo" ./tasks.py
sudo systemctl stop redis-server && redis-server --bind 0.0.0.0 --protected-mode no