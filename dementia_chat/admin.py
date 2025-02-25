from django.contrib import admin
from .models import Utterance
# Register your models here.

@admin.register(Utterance)
class UtteranceAdmin(admin.ModelAdmin):
    list_display = ('speaker', 'text', 'timestamp', 'session_id')
    list_filter = ('speaker', 'timestamp')
    search_fields = ('text', 'session_id')
    date_hierarchy = 'timestamp'

