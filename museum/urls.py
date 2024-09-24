# museum/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.welcome, name='welcome'),
    path('signup/', views.signup, name='signup'),
    path('login/', views.login_view, name='login'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('add_museum/', views.add_museum, name='add_museum'),
    path('add_rooms/', views.add_rooms, name='add_rooms'),
    path('add_cctv_feed/', views.add_cctv_feed, name='add_cctv_feed'),
    path('add_cctv_footages/', views.add_cctv_footages, name='add_cctv_footages'),
    path('list_cctv/', views.list_cctv, name='list_cctv'),
    path('draw_roi/<int:feed_id>/', views.draw_roi, name='draw_roi'),
    path('process_video_feed/<int:feed_id>/', views.process_video_feed, name='process_video_feed'),
    path('logout/', views.logout_view, name='logout'),
    path('run_detection/', views.run_detection, name='run_detection'),
    path('stream_video/<int:feed_id>/', views.stream_video, name='stream_video_feed'),
    path('stream_video/footage/<int:footage_id>/', views.stream_video, name='stream_video_footage'),
    path('view_results/', views.view_results, name='view_results'),
    path('api/get_feeds_and_footages/', views.get_feeds_and_footages, name='get_feeds_and_footages'),
]