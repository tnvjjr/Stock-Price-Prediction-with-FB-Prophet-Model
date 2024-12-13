"""
WSGI config for stock_prediction project.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_prediction.settings')

application = get_wsgi_application()
