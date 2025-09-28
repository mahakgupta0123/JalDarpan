#!/bin/sh
exec gunicorn --bind 0.0.0.0:${PORT:-8000} flask_app:app