# Generated by Django 3.2.25 on 2025-02-25 04:11

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Utterance',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('speaker', models.CharField(choices=[('User', 'User'), ('System', 'System')], max_length=10)),
                ('text', models.TextField()),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('session_id', models.CharField(max_length=100)),
            ],
            options={
                'ordering': ['timestamp'],
            },
        ),
    ]
