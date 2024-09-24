# museum/forms.py

from django import forms
from .models import Museum, Room, CCTVFeed

class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True

class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = [single_file_clean(data, initial)]
        return result

class MuseumForm(forms.ModelForm):
    class Meta:
        model = Museum
        fields = ['owner', 'museum_name']

class RoomForm(forms.ModelForm):
    class Meta:
        model = Room
        fields = ['museum_name', 'room_name']

class CCTVFeedForm(forms.ModelForm):
    class Meta:
        model = CCTVFeed
        fields = ['room_name', 'cctv_name', 'feed_url']

class CCTVFootagesForm(forms.Form):
    cctv_name = forms.ModelChoiceField(queryset=CCTVFeed.objects.all())
    video_files = MultipleFileField()