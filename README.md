# Real-Time Fall Detection and Action Recognition in Video

Graduation project 2026-2027. Supervisor: Prof. Usama Sayed.

## Pipeline
video or camera -> YOLOv8 (people) -> tracking (IDs) -> preprocessing -> action model -> fall decision -> alert

## Folders
- detection/      YOLOv8 module, fall rule, fake pipeline
- tracking/       tracker
- recognition/    action recognition model
- preprocessing/  preprocessing and data loader
- frontend/       web interface
- server/         small API that runs the pipeline
- docs/           BRS, SRS, reports
- data/, models/, test_videos/   large files, NOT in Git

## Team
Yassien Islam (leader, detection), Naira, Hussien Essam, Sohaila Abdel Aall, Hesham Abd Elghany, Arwa Mostafa
