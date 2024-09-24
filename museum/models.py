# museum/models.py

from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone

class Museum(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    museum_name = models.CharField(max_length=255)

    def __str__(self):
        return self.museum_name

class Room(models.Model):
    museum_name = models.ForeignKey(Museum, on_delete=models.CASCADE)
    room_name = models.CharField(max_length=255)

    def __str__(self):
        return self.room_name

class CCTVFeed(models.Model):
    room_name = models.ForeignKey(Room, on_delete=models.CASCADE)
    cctv_name = models.CharField(max_length=255)
    feed_url = models.CharField(max_length=255)
    first_frame_image = models.ImageField(upload_to='first_frames/', blank=True, null=True)

    def __str__(self):
        return self.cctv_name

class CCTVFootages(models.Model):
    cctv_name = models.ForeignKey(CCTVFeed, on_delete=models.CASCADE)
    video_file = models.FileField(upload_to='videos/', blank=True, null=True)

class ROI(models.Model):
    cctv_name = models.ForeignKey(CCTVFeed, on_delete=models.CASCADE)
    roi_name = models.CharField(max_length=255)
    coordinates = models.TextField()

class DetectionResults(models.Model):
    roi_name = models.ForeignKey(ROI, on_delete=models.CASCADE)
    room = models.ForeignKey(Room, on_delete=models.CASCADE, null=True)
    date = models.DateField()
    time_start = models.TimeField(default=timezone.now)
    time_end = models.TimeField(default=timezone.now)
    visitor_passing_count = models.IntegerField(default=0)
    visitor_interested_count = models.IntegerField(default=0)