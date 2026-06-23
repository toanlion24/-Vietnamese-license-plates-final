# Cursor Rules Index

This file enables Cursor to automatically load all rules for the Vietnamese LPR project.

## Rules

| File | Purpose |
|------|---------|
| `vietnamese-lpr.md` | Main project context and conventions |
| `data-processing.md` | Dataset preparation and augmentation |
| `model-training.md` | Training configuration and best practices |
| `debugging.md` | Debugging guide and troubleshooting |
| `deployment.md` | Deployment options and Docker |
| `best-practices.md` | Code organization and testing |

## Modules

| File | Purpose |
|------|---------|
| `modules/input_stream.py` | Module 1: Input Stream (Image/Video/Webcam) |
| `modules/yolo_detection.py` | Module 2: YOLOv11 Detection |
| `modules/vehicle_plate_association.py` | Module 3: Vehicle-Plate Association |
| `modules/rectify_perspective.py` | Module 4: Rectify & Perspective Correction |
| `modules/image_enhancement.py` | Module 5: Image Enhancement (CLAHE, denoise) |
| `modules/paddleocr_extraction.py` | Module 6: PaddleOCR Character Recognition |
| `modules/rule_engine.py` | Module 7: Rule Engine & Regex Validation |
| `modules/bytetrack_voting.py` | Module 8: ByteTrack & Voting System |
| `modules/database_manager.py` | Module 9: Database Storage |
| `ui/streamlit_app.py` | Streamlit Web Interface |

## Skills

| File | Purpose |
|------|---------|
| `skills/vietnamese-lpr-pipeline.md` | Pipeline implementation |
| `skills/vietnamese-lpr-detection.md` | Detection module |
| `skills/vietnamese-lpr-recognition.md` | Recognition module |
| `skills/vietnamese-lpr-evaluation.md` | Evaluation and benchmarking |

## How Rules Work

Rules are automatically loaded by Cursor when you open the project. They provide:

1. **Context** - What the project is about
2. **Conventions** - Coding standards
3. **Patterns** - Common solutions
4. **Constraints** - What to avoid

## How Skills Work

Skills are activated by:
- Skill commands (if defined)
- Auto-detection based on file patterns
- Explicit mention in conversation

Example skill activation:
```
Use the [vietnamese-lpr-pipeline] skill to implement this feature.
```

## Auto-activation Rules

Skills auto-activate when you work on:

| Pattern | Skill |
|---------|-------|
| `modules/input_stream.py` | Pipeline (Input) |
| `modules/yolo_detection.py` | Detection |
| `modules/vehicle_plate_association.py` | Pipeline (Association) |
| `modules/rectify_perspective.py` | Pipeline (Preprocessing) |
| `modules/image_enhancement.py` | Pipeline (Enhancement) |
| `modules/paddleocr_extraction.py` | Recognition |
| `modules/rule_engine.py` | Recognition (Validation) |
| `modules/bytetrack_voting.py` | Pipeline (Tracking) |
| `modules/database_manager.py` | Pipeline (Storage) |
| `ui/streamlit_app.py` | UI Development |

## Customization

To modify rules:
1. Edit the rule file directly
2. Changes take effect immediately
3. Cursor will use updated rules in next session

To create new skills:
1. Create `skills/your-skill-name.md`
2. Follow the skill template format
3. Add to this index if auto-activation desired
