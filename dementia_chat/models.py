from django.db import models

class Utterance(models.Model):
    SPEAKER_CHOICES = [
        ('User', 'User'),
        ('System', 'System'),
    ]
    
    speaker = models.CharField(max_length=10, choices=SPEAKER_CHOICES)
    text = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    session_id = models.CharField(max_length=100)

    class Meta:
        app_label = 'dementia_chat'

    def __str__(self):
        return f"{self.speaker}: {self.text[:50]}..."

