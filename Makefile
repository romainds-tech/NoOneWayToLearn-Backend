RSYNC_CMD = rsync -avz --delete --no-perms
RSYNC_CMD += --exclude '.git*'
RSYNC_CMD += --exclude config.py
RSYNC_CMD += --exclude Makefile
RSYNC_CMD += --exclude '*.db'
RSYNC_CMD += --exclude '*.pyc'
RSYNC_CMD += --exclude .venv
RSYNC_CMD += --exclude app.py
RSYNC_CMD += --exclude loggs

dev:
	${RSYNC_CMD} . user@domain:~/repertoire/
