#!/usr/bin/env python3
"""
Omni-Shield Synthetic ID Card Generator
========================================
Stage 1 of the knowledge distillation pipeline.

Generates realistic synthetic identity documents for 6 countries:
  - USA: Driver's Licence (AAMVA format)
  - India: Aadhaar Card + PAN Card
  - UK: Driving Licence (DVLA format)
  - Germany: Personalausweis (national ID)
  - Australia: Driver's Licence
  - Canada: Driver's Licence

Each generated image comes with a ground_truth.json file containing
the exact bounding box of every PII field — because we control the
rendering, we know precisely where everything is. This gives us perfect
annotations for free, which we then verify and enrich using Claude
in Stage 2 (annotate_with_claude.py).

The combination of synthetic images + ground truth annotations forms
the training dataset for fine-tuning Qwen2-VL-2B to become a specialist
PII detector for government identity documents.

Usage:
    pip install Pillow faker numpy
    python generate_synthetic_ids.py --count 500 --output_dir ./dataset

Output structure:
    dataset/
      images/          ← PNG images, one per card
      annotations/     ← JSON ground truth, one per card
      manifest.json    ← list of all generated files + metadata
"""

import argparse
import json
import os
import random
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from faker import Faker

# ── Faker instances for each locale ───────────────────────────────────────────
# Each locale gives us realistic names, addresses, and numbers for that region.
FAKERS = {
    "USA":       Faker("en_US"),
    "India":     Faker("en_IN"),
    "UK":        Faker("en_GB"),
    "Germany":   Faker("de_DE"),
    "Australia": Faker("en_AU"),
    "Canada":    Faker("en_CA"),
}

# Seed for reproducibility — remove or change for truly random output
random.seed(42)
for f in FAKERS.values():
    f.seed_instance(42)


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class BoundingBox:
    """
    Pixel-space bounding box [xmin, ymin, xmax, ymax].
    Also stores the normalized 0-1000 version for VLM training.
    """
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    # These are filled in after we know the image dimensions
    norm_ymin: int = 0
    norm_xmin: int = 0
    norm_ymax: int = 0
    norm_xmax: int = 0

    def normalize(self, img_width: int, img_height: int):
        self.norm_ymin = int((self.ymin / img_height) * 1000)
        self.norm_xmin = int((self.xmin / img_width)  * 1000)
        self.norm_ymax = int((self.ymax / img_height) * 1000)
        self.norm_xmax = int((self.xmax / img_width)  * 1000)
        return self


@dataclass
class PIIField:
    """One annotated PII field in a document."""
    label: str           # human-readable field name e.g. "Full Name"
    value: str           # the actual text value e.g. "John Smith"
    is_pii: bool         # always True for this dataset
    box: BoundingBox     # where it appears on the image
    field_type: str = "text"  # "text" | "face" | "signature" | "number"


@dataclass
class CardAnnotation:
    """Complete annotation for one generated card."""
    image_id: str
    country: str
    card_type: str
    image_path: str
    image_width: int
    image_height: int
    fields: list[PIIField] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        return d


# ══════════════════════════════════════════════════════════════════════════════
# FONT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """
    Load a system font. Falls back to PIL's built-in bitmap font if
    no system fonts are available — the bitmap font is ugly but always works.
    """
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",   # macOS fallback
        "C:/Windows/Fonts/arial.ttf",             # Windows fallback
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    """Return (width, height) of rendered text."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC FACE PLACEHOLDER
# Instead of real faces (which raise privacy and copyright concerns), we
# generate a stylized placeholder rectangle that clearly represents a photo
# zone. This is sufficient for teaching the model "there is a face here."
# ══════════════════════════════════════════════════════════════════════════════
def _draw_face_placeholder(
    draw: ImageDraw.ImageDraw,
    x: int, y: int, w: int, h: int,
    skin_tone: tuple = (210, 180, 140),
) -> None:
    """Draw a stylized placeholder representing a person photograph."""
    # Background — cream/off-white like a photo background
    draw.rectangle([x, y, x+w, y+h], fill=(245, 245, 240), outline=(180, 180, 180), width=1)

    # Head shape (oval)
    head_x = x + w//4
    head_y = y + h//8
    head_w = w//2
    head_h = int(h * 0.45)
    draw.ellipse([head_x, head_y, head_x+head_w, head_y+head_h], fill=skin_tone)

    # Body/shoulders
    shoulder_y = y + int(h * 0.6)
    draw.ellipse(
        [x + w//8, shoulder_y, x + w - w//8, y + h + h//3],
        fill=tuple(max(0, c-30) for c in skin_tone)
    )

    # Simple eyes
    eye_y = head_y + head_h//3
    eye_offset = head_w//5
    for ex in [head_x + eye_offset, head_x + head_w - eye_offset]:
        draw.ellipse([ex-3, eye_y-3, ex+3, eye_y+3], fill=(50, 30, 10))

    # Border to make it clearly a photo zone
    draw.rectangle([x, y, x+w, y+h], outline=(100, 100, 100), width=2)


def _draw_signature_placeholder(
    draw: ImageDraw.ImageDraw,
    x: int, y: int, w: int, h: int,
    name: str,
) -> None:
    """
    Draw a wavy line sequence that represents a handwritten signature.
    We use the name to seed the wave pattern so different names produce
    visually different signatures.
    """
    # Seed the wave using a hash of the name
    seed = sum(ord(c) for c in name) % 1000
    rng = random.Random(seed)

    # Draw a baseline
    baseline_y = y + h * 2 // 3
    points = []
    cx = x + 5
    while cx < x + w - 5:
        # Wavy cursive-like strokes
        wave_y = baseline_y + rng.randint(-h//3, h//4)
        points.append((cx, wave_y))
        cx += rng.randint(3, 8)

    if len(points) >= 2:
        draw.line(points, fill=(20, 20, 80), width=2)

    # A few upward loops to suggest capital letters
    for _ in range(3):
        lx = x + rng.randint(10, w-20)
        ly = baseline_y - rng.randint(5, h//2)
        draw.arc([lx, ly, lx+10, baseline_y], 0, 270, fill=(20, 20, 80), width=2)


# ══════════════════════════════════════════════════════════════════════════════
# CARD GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def _add_noise_and_effects(img: Image.Image, skew_angle: float = 0.0) -> Image.Image:
    """
    Add realistic degradation to a synthetic card:
    - Slight JPEG compression artifacts
    - Mild Gaussian blur (simulating camera focus variation)
    - Random brightness/contrast shift (simulating lighting variation)
    - Optional skew (simulating non-flat placement)
    """
    import io

    # Random brightness variation ±15%
    factor = random.uniform(0.85, 1.15)
    arr = np.array(img, dtype=np.float32)
    arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Mild blur — simulates slight defocus
    if random.random() < 0.4:
        radius = random.uniform(0.3, 0.8)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Simulate JPEG compression (quality 75-95)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=random.randint(75, 95))
    buf.seek(0)
    img = Image.open(buf).copy()

    # Optional skew
    if abs(skew_angle) > 0.5:
        img = img.rotate(skew_angle, expand=True, fillcolor=(255, 255, 255))

    return img


def generate_usa_drivers_licence(output_dir: Path) -> CardAnnotation:
    """
    Generate a US driver's licence in the AAMVA standard visual format.
    Fields: name (surname + given), DOB, address, city, state, zip,
    DL number, expiry, issue date, sex, eyes, height, DD number, face, signature.
    """
    fake = FAKERS["USA"]
    W, H = 856, 540   # Standard ID-1 credit card size at 96dpi

    # ── Generate PII values ────────────────────────────────────────────────────
    first_name  = fake.first_name()
    last_name   = fake.last_name()
    full_name   = f"{last_name}, {first_name}"
    dob         = fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%m/%d/%Y")
    address     = fake.street_address().upper()
    city        = fake.city().upper()
    state_abbr  = fake.state_abbr()
    zip_code    = fake.zipcode()
    dl_number   = f"{state_abbr}{fake.numerify('########')}"
    expiry      = fake.date_between(start_date="+1y", end_date="+5y").strftime("%m/%d/%Y")
    issue_date  = fake.date_between(start_date="-5y", end_date="today").strftime("%m/%d/%Y")
    sex         = random.choice(["M", "F"])
    eyes        = random.choice(["BRO", "BLU", "GRN", "HAZ", "GRY"])
    height      = f"{random.randint(4, 6)}'{random.randint(0, 11):02d}\""
    dd_number   = fake.numerify("####################")
    state_name  = fake.state()

    # ── Create image ──────────────────────────────────────────────────────────
    img = Image.new("RGB", (W, H), (245, 245, 230))   # off-white card background
    draw = ImageDraw.ImageDraw(img)

    # Header bar — state colour (gold/blue)
    draw.rectangle([0, 0, W, 60], fill=(0, 56, 117))   # Pennsylvania-style blue
    draw.rectangle([0, 60, W, 80], fill=(230, 180, 0))  # gold stripe

    # State name
    f_state = _font(28, bold=True)
    draw.text((20, 15), state_name.upper(), fill=(255, 255, 255), font=f_state)
    draw.text((20, 65), "DRIVER LICENSE", fill=(0, 0, 0), font=_font(10, bold=True))
    draw.text((W-160, 15), "USA", fill=(200, 200, 200), font=_font(22))

    # Tracking variables for annotation
    fields: list[PIIField] = []
    image_id = str(uuid.uuid4())

    def _field(draw, x, y, label_text, value_text, font_size=13,
               label_font_size=9, color=(0,0,0), bold_value=False):
        """Helper: draw a label+value pair and return its annotation."""
        lf = _font(label_font_size)
        vf = _font(font_size, bold=bold_value)
        # Draw label
        draw.text((x, y), label_text, fill=(80, 80, 80), font=lf)
        lw, lh = _text_size(draw, label_text, lf)
        # Draw value below label
        vy = y + lh + 1
        draw.text((x, vy), value_text, fill=color, font=vf)
        vw, vh = _text_size(draw, value_text, vf)
        box = BoundingBox(xmin=x, ymin=vy, xmax=x+vw, ymax=vy+vh)
        return box

    # ── Face photograph zone ──────────────────────────────────────────────────
    face_x, face_y, face_w, face_h = 20, 90, 160, 200
    _draw_face_placeholder(draw, face_x, face_y, face_w, face_h)
    face_box = BoundingBox(xmin=face_x, ymin=face_y, xmax=face_x+face_w, ymax=face_y+face_h)
    fields.append(PIIField("Face", "face", True, face_box, "face"))

    # ── DL Number (top right, prominent) ─────────────────────────────────────
    dl_x = W - 250
    draw.text((dl_x, 90), "4d DL", fill=(60, 60, 60), font=_font(9))
    draw.text((dl_x, 102), dl_number, fill=(0, 0, 0), font=_font(18, bold=True))
    _, lh = _text_size(draw, "4d DL", _font(9))
    dw, dh = _text_size(draw, dl_number, _font(18, bold=True))
    fields.append(PIIField("ID Number", dl_number, True,
        BoundingBox(dl_x, 102, dl_x+dw, 102+dh), "number"))

    # ── DOB ───────────────────────────────────────────────────────────────────
    box = _field(draw, dl_x, 128, "3 DOB", dob, font_size=15, bold_value=True)
    fields.append(PIIField("Date of Birth", dob, True, box, "text"))

    # ── Name ──────────────────────────────────────────────────────────────────
    name_x, name_y = 195, 90
    draw.text((name_x, name_y), "2 NAME", fill=(60,60,60), font=_font(9))
    draw.text((name_x, name_y+11), full_name, fill=(0,0,0), font=_font(16, bold=True))
    nw, nh = _text_size(draw, full_name, _font(16, bold=True))
    fields.append(PIIField("Full Name", full_name, True,
        BoundingBox(name_x, name_y+11, name_x+nw, name_y+11+nh), "text"))

    # ── Address ───────────────────────────────────────────────────────────────
    addr_y = name_y + 40
    draw.text((name_x, addr_y), "8 ADDR", fill=(60,60,60), font=_font(9))
    draw.text((name_x, addr_y+11), address, fill=(0,0,0), font=_font(12))
    aw, ah = _text_size(draw, address, _font(12))
    fields.append(PIIField("Address", address, True,
        BoundingBox(name_x, addr_y+11, name_x+aw, addr_y+11+ah), "text"))

    city_state = f"{city}, {state_abbr} {zip_code}"
    draw.text((name_x, addr_y+26), city_state, fill=(0,0,0), font=_font(12))
    csw, csh = _text_size(draw, city_state, _font(12))
    fields.append(PIIField("City State Zip", city_state, True,
        BoundingBox(name_x, addr_y+26, name_x+csw, addr_y+26+csh), "text"))

    # ── Expiry, Issue ─────────────────────────────────────────────────────────
    draw.text((name_x, addr_y+50), f"4b EXP: {expiry}", fill=(60,60,60), font=_font(12))
    draw.text((name_x, addr_y+66), f"4a ISS: {issue_date}", fill=(60,60,60), font=_font(12))

    # ── Sex, Eyes, Height ────────────────────────────────────────────────────
    misc_y = addr_y + 90
    draw.text((name_x, misc_y),
        f"15 SEX: {sex}   18 EYES: {eyes}   16 HGT: {height}",
        fill=(60,60,60), font=_font(11))

    # ── DD Number (document discriminator) ───────────────────────────────────
    dd_y = H - 50
    draw.text((20, dd_y), "5 DD:", fill=(80,80,80), font=_font(9))
    draw.text((20, dd_y+12), dd_number[:20], fill=(40,40,40), font=_font(10))
    draw.text((20, dd_y+24), dd_number[20:], fill=(40,40,40), font=_font(10))
    ddn_box = BoundingBox(20, dd_y+12, 20+len(dd_number[:20])*6, dd_y+36)
    fields.append(PIIField("Document Discriminator", dd_number, True, ddn_box, "number"))

    # ── Signature zone ───────────────────────────────────────────────────────
    sig_x, sig_y, sig_w, sig_h = 20, 305, 160, 40
    draw.rectangle([sig_x, sig_y, sig_x+sig_w, sig_y+sig_h],
                   outline=(150,150,150), width=1)
    _draw_signature_placeholder(draw, sig_x+5, sig_y+2, sig_w-10, sig_h-4,
                                 f"{first_name} {last_name}")
    sig_box = BoundingBox(sig_x, sig_y, sig_x+sig_w, sig_y+sig_h)
    fields.append(PIIField("Signature", f"{first_name} {last_name}", True, sig_box, "signature"))

    # ── Border ───────────────────────────────────────────────────────────────
    draw.rectangle([0, 0, W-1, H-1], outline=(80, 80, 80), width=3)

    # ── Apply effects ─────────────────────────────────────────────────────────
    skew = random.uniform(-5, 5) if random.random() < 0.3 else 0.0
    img = _add_noise_and_effects(img, skew_angle=skew)
    final_w, final_h = img.size

    # Normalize all bounding boxes to 0-1000
    for f_item in fields:
        f_item.box.normalize(final_w, final_h)

    # Save image
    fname = f"usa_dl_{image_id}.png"
    img_path = output_dir / "images" / fname
    img.save(img_path)

    return CardAnnotation(
        image_id=image_id,
        country="USA",
        card_type="Driver's Licence",
        image_path=str(img_path),
        image_width=final_w,
        image_height=final_h,
        fields=fields,
    )


def generate_india_aadhaar(output_dir: Path) -> CardAnnotation:
    """
    Generate a synthetic Indian Aadhaar card.
    Fields: name, DOB, gender, Aadhaar number (12 digits), address, face.
    """
    fake = FAKERS["India"]
    W, H = 856, 540

    first   = fake.first_name()
    last    = fake.last_name()
    name    = f"{first} {last}"
    dob     = fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%d/%m/%Y")
    gender  = random.choice(["MALE", "FEMALE"])
    # Aadhaar: 12 digits formatted as XXXX XXXX XXXX
    raw     = fake.numerify("############")
    aadhaar = f"{raw[:4]} {raw[4:8]} {raw[8:]}"
    address = f"{fake.street_address()}, {fake.city()}, {fake.state()}"

    img  = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.ImageDraw(img)

    # Header — UIDAI blue
    draw.rectangle([0, 0, W, 70], fill=(0, 84, 166))
    draw.text((20, 10), "आधार", fill=(255,255,255), font=_font(30, bold=True))
    draw.text((20, 42), "AADHAAR", fill=(255,200,0), font=_font(18, bold=True))
    draw.text((W-200, 10), "भारत सरकार", fill=(220,220,220), font=_font(14))
    draw.text((W-200, 30), "Government of India", fill=(200,200,200), font=_font(12))

    # Orange stripe at bottom
    draw.rectangle([0, H-60, W, H], fill=(255, 153, 51))
    draw.text((20, H-45), "Unique Identification Authority of India",
              fill=(255,255,255), font=_font(11))

    fields: list[PIIField] = []
    image_id = str(uuid.uuid4())

    # Face
    face_x, face_y, face_w, face_h = 20, 85, 140, 180
    _draw_face_placeholder(draw, face_x, face_y, face_w, face_h, skin_tone=(180, 130, 100))
    fields.append(PIIField("Face", "face", True,
        BoundingBox(face_x, face_y, face_x+face_w, face_y+face_h).normalize(W, H), "face"))

    # Name
    tx = 175
    draw.text((tx, 90), "Name", fill=(100,100,100), font=_font(10))
    draw.text((tx, 103), name, fill=(0,0,0), font=_font(18, bold=True))
    nw, nh = _text_size(draw, name, _font(18, bold=True))
    fields.append(PIIField("Full Name", name, True,
        BoundingBox(tx, 103, tx+nw, 103+nh).normalize(W, H), "text"))

    # DOB
    draw.text((tx, 130), "Date of Birth", fill=(100,100,100), font=_font(10))
    draw.text((tx, 143), dob, fill=(0,0,0), font=_font(15))
    dw, dh = _text_size(draw, dob, _font(15))
    fields.append(PIIField("Date of Birth", dob, True,
        BoundingBox(tx, 143, tx+dw, 143+dh).normalize(W, H), "text"))

    # Gender
    draw.text((tx, 165), "Gender", fill=(100,100,100), font=_font(10))
    draw.text((tx, 178), gender, fill=(0,0,0), font=_font(13))

    # Address
    draw.text((tx, 200), "Address", fill=(100,100,100), font=_font(10))
    addr_lines = [address[i:i+45] for i in range(0, len(address), 45)]
    for i, line in enumerate(addr_lines[:3]):
        ay = 213 + i * 16
        draw.text((tx, ay), line, fill=(40,40,40), font=_font(11))
    if addr_lines:
        first_line = addr_lines[0]
        lw, lh = _text_size(draw, first_line, _font(11))
        total_h = len(addr_lines[:3]) * 16
        fields.append(PIIField("Address", address, True,
            BoundingBox(tx, 213, tx+lw, 213+total_h).normalize(W, H), "text"))

    # Aadhaar number — large, centered, prominent
    an_y = H - 110
    draw.text((20, an_y), "Your Aadhaar No.", fill=(80,80,80), font=_font(11))
    draw.text((20, an_y+16), aadhaar, fill=(0,0,0), font=_font(28, bold=True))
    aw, ah = _text_size(draw, aadhaar, _font(28, bold=True))
    fields.append(PIIField("Aadhaar Number", aadhaar, True,
        BoundingBox(20, an_y+16, 20+aw, an_y+16+ah).normalize(W, H), "number"))

    # Signature
    sig_x, sig_y, sig_w, sig_h = W-200, H-110, 180, 50
    draw.text((sig_x, sig_y-12), "Signature", fill=(80,80,80), font=_font(10))
    draw.rectangle([sig_x, sig_y, sig_x+sig_w, sig_y+sig_h],
                   outline=(180,180,180), width=1)
    _draw_signature_placeholder(draw, sig_x+5, sig_y+5, sig_w-10, sig_h-10, name)
    fields.append(PIIField("Signature", name, True,
        BoundingBox(sig_x, sig_y, sig_x+sig_w, sig_y+sig_h).normalize(W, H), "signature"))

    # Border
    draw.rectangle([0, 0, W-1, H-1], outline=(0, 84, 166), width=3)

    skew = random.uniform(-5, 5) if random.random() < 0.3 else 0.0
    img = _add_noise_and_effects(img, skew_angle=skew)
    final_w, final_h = img.size

    fname = f"india_aadhaar_{image_id}.png"
    img_path = output_dir / "images" / fname
    img.save(img_path)

    return CardAnnotation(
        image_id=image_id, country="India", card_type="Aadhaar Card",
        image_path=str(img_path), image_width=final_w, image_height=final_h,
        fields=fields,
    )


def generate_uk_driving_licence(output_dir: Path) -> CardAnnotation:
    """Generate a synthetic UK DVLA driving licence."""
    fake = FAKERS["UK"]
    W, H = 856, 540

    surname   = fake.last_name().upper()
    given     = fake.first_name()
    dob       = fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%d.%m.%Y")
    place_birth = fake.city().upper()
    expiry    = fake.date_between(start_date="+1y", end_date="+10y").strftime("%d.%m.%Y")
    issue_date = fake.date_between(start_date="-5y", end_date="today").strftime("%d.%m.%Y")
    dl_number = f"{surname[:5].upper():5}{fake.numerify('9######')}{given[:2].upper()}9{fake.numerify('##')}"
    address   = fake.street_address()
    postcode  = fake.postcode()
    city      = fake.city()

    img  = Image.new("RGB", (W, H), (240, 240, 220))
    draw = ImageDraw.ImageDraw(img)

    # Green header
    draw.rectangle([0, 0, W, 65], fill=(0, 100, 0))
    draw.text((20, 8),  "DRIVING LICENCE", fill=(255,255,255), font=_font(22, bold=True))
    draw.text((20, 36), "UNITED KINGDOM", fill=(200,255,200), font=_font(14))
    # EU-style circle (simplified)
    draw.ellipse([W-65, 8, W-10, 55], outline=(255,200,0), width=3)
    draw.text((W-52, 20), "UK", fill=(255,255,255), font=_font(14, bold=True))

    fields: list[PIIField] = []
    image_id = str(uuid.uuid4())

    # Face
    face_x, face_y, face_w, face_h = 20, 80, 145, 185
    _draw_face_placeholder(draw, face_x, face_y, face_w, face_h)
    fields.append(PIIField("Face", "face", True,
        BoundingBox(face_x, face_y, face_x+face_w, face_y+face_h).normalize(W, H), "face"))

    # Fields (DVLA numbering)
    tx = 180

    def dvla_field(num, label, value, y, font_size=13, is_pii_flag=True, bold=False):
        draw.text((tx, y), f"{num}. {label}", fill=(80,80,80), font=_font(9))
        draw.text((tx, y+12), value, fill=(0,0,0), font=_font(font_size, bold=bold))
        vw, vh = _text_size(draw, value, _font(font_size, bold=bold))
        box = BoundingBox(tx, y+12, tx+vw, y+12+vh).normalize(W, H)
        if is_pii_flag:
            fields.append(PIIField(label, value, True, box, "text"))

    dvla_field("1",  "Surname",     surname,      90,  font_size=16, bold=True)
    dvla_field("2",  "First names", given,         120, font_size=14)
    dvla_field("3",  "Date of birth", dob,         150, font_size=14)
    dvla_field("4b", "Expiry",      expiry,        180, is_pii_flag=False)
    dvla_field("4c", "Issue date",  issue_date,    208, is_pii_flag=False)
    dvla_field("4d", "Licence No.", dl_number,     240, font_size=15, bold=True)
    dvla_field("8",  "Address",     f"{address}, {city}", 275)
    dvla_field("",   "Postcode",    postcode,      303)

    # Signature
    sig_x, sig_y, sig_w, sig_h = tx, 330, 220, 45
    draw.text((sig_x, sig_y-12), "Signature", fill=(80,80,80), font=_font(9))
    draw.rectangle([sig_x, sig_y, sig_x+sig_w, sig_y+sig_h],
                   outline=(160,160,160), width=1)
    _draw_signature_placeholder(draw, sig_x+5, sig_y+3, sig_w-10, sig_h-6,
                                 f"{given} {surname}")
    fields.append(PIIField("Signature", f"{given} {surname}", True,
        BoundingBox(sig_x, sig_y, sig_x+sig_w, sig_y+sig_h).normalize(W, H), "signature"))

    draw.rectangle([0, 0, W-1, H-1], outline=(0,100,0), width=3)

    skew = random.uniform(-5, 5) if random.random() < 0.3 else 0.0
    img = _add_noise_and_effects(img, skew_angle=skew)
    final_w, final_h = img.size

    fname = f"uk_dl_{image_id}.png"
    img_path = output_dir / "images" / fname
    img.save(img_path)

    return CardAnnotation(
        image_id=image_id, country="UK", card_type="Driving Licence",
        image_path=str(img_path), image_width=final_w, image_height=final_h,
        fields=fields,
    )


def generate_germany_personalausweis(output_dir: Path) -> CardAnnotation:
    """Generate a synthetic German Personalausweis (national identity card)."""
    fake = FAKERS["Germany"]
    W, H = 856, 540

    surname   = fake.last_name().upper()
    given     = fake.first_name()
    dob       = fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%d.%m.%Y")
    pob       = fake.city().upper()
    expiry    = fake.date_between(start_date="+1y", end_date="+10y").strftime("%d.%m.%Y")
    nationality = "DEUTSCH / GERMAN"
    doc_no    = fake.bothify("?#########").upper()
    address   = fake.street_address()
    city      = fake.city()

    img  = Image.new("RGB", (W, H), (248, 248, 240))
    draw = ImageDraw.ImageDraw(img)

    # Black/red/gold header (German flag colours)
    draw.rectangle([0, 0, W, 22], fill=(0, 0, 0))
    draw.rectangle([0, 22, W, 44], fill=(180, 0, 0))
    draw.rectangle([0, 44, W, 66], fill=(200, 160, 0))
    draw.text((W//2-100, 4),  "BUNDESREPUBLIK DEUTSCHLAND",
              fill=(255,255,255), font=_font(12, bold=True))
    draw.text((W//2-80, 26), "PERSONALAUSWEIS",
              fill=(255,255,255), font=_font(14, bold=True))
    draw.text((W//2-60, 49), "IDENTITY CARD",
              fill=(0,0,0), font=_font(12))

    fields: list[PIIField] = []
    image_id = str(uuid.uuid4())

    # Face
    face_x, face_y, face_w, face_h = 20, 80, 140, 180
    _draw_face_placeholder(draw, face_x, face_y, face_w, face_h)
    fields.append(PIIField("Face", "face", True,
        BoundingBox(face_x, face_y, face_x+face_w, face_y+face_h).normalize(W, H), "face"))

    tx = 175

    def de_field(label_de, label_en, value, y, font_size=13, pii=True, bold=False):
        draw.text((tx, y), f"{label_de} / {label_en}", fill=(80,80,80), font=_font(8))
        draw.text((tx, y+11), value, fill=(0,0,0), font=_font(font_size, bold=bold))
        vw, vh = _text_size(draw, value, _font(font_size, bold=bold))
        box = BoundingBox(tx, y+11, tx+vw, y+11+vh).normalize(W, H)
        if pii:
            fields.append(PIIField(label_en, value, True, box, "text"))

    de_field("Name", "Surname", surname, 80, font_size=16, bold=True)
    de_field("Vornamen", "Given Names", given, 112, font_size=14)
    de_field("Geburtsdatum", "Date of Birth", dob, 144)
    de_field("Geburtsort", "Place of Birth", pob, 172, pii=False)
    de_field("Staatsangehörigkeit", "Nationality", nationality, 200, pii=False)
    de_field("Gültig bis", "Expiry", expiry, 228, pii=False)
    de_field("Ausweisnummer", "Document Number", doc_no, 256, bold=True)
    de_field("Anschrift", "Address", f"{address}, {city}", 284)

    # Signature
    sig_x, sig_y, sig_w, sig_h = tx, 320, 200, 45
    draw.text((sig_x, sig_y-12), "Unterschrift / Signature", fill=(80,80,80), font=_font(9))
    draw.rectangle([sig_x, sig_y, sig_x+sig_w, sig_y+sig_h],
                   outline=(160,160,160), width=1)
    _draw_signature_placeholder(draw, sig_x+5, sig_y+3, sig_w-10, sig_h-6,
                                 f"{given} {surname}")
    fields.append(PIIField("Signature", f"{given} {surname}", True,
        BoundingBox(sig_x, sig_y, sig_x+sig_w, sig_y+sig_h).normalize(W, H), "signature"))

    # MRZ strip at bottom
    draw.rectangle([0, H-55, W, H], fill=(220, 220, 200))
    mrz1 = f"IDD<<{surname}<<{given}<<<<<<<<<<<<<<<<<<<"[:44]
    mrz2 = f"{doc_no}D<<{dob[6:]}{dob[3:5]}{dob[:2]}0{expiry[6:]}{expiry[3:5]}{expiry[:2]}0<<<<<<<<<<<<<<<<"[:44]
    draw.text((10, H-52), mrz1, fill=(0,0,0), font=_font(11))
    draw.text((10, H-35), mrz2, fill=(0,0,0), font=_font(11))

    draw.rectangle([0, 0, W-1, H-1], outline=(0,0,0), width=2)

    skew = random.uniform(-5, 5) if random.random() < 0.3 else 0.0
    img = _add_noise_and_effects(img, skew_angle=skew)
    final_w, final_h = img.size

    fname = f"germany_id_{image_id}.png"
    img_path = output_dir / "images" / fname
    img.save(img_path)

    return CardAnnotation(
        image_id=image_id, country="Germany", card_type="Personalausweis",
        image_path=str(img_path), image_width=final_w, image_height=final_h,
        fields=fields,
    )


def generate_australia_drivers_licence(output_dir: Path) -> CardAnnotation:
    """Generate a synthetic Australian driver's licence."""
    fake = FAKERS["Australia"]
    W, H = 856, 540

    first   = fake.first_name()
    last    = fake.last_name().upper()
    dob     = fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%d/%m/%Y")
    expiry  = fake.date_between(start_date="+1y", end_date="+5y").strftime("%d/%m/%Y")
    licence = fake.bothify("??#######").upper()
    address = fake.street_address()
    suburb  = fake.city().upper()
    state   = random.choice(["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"])
    postcode = fake.postcode()

    img  = Image.new("RGB", (W, H), (245, 242, 230))
    draw = ImageDraw.ImageDraw(img)

    # State header colour map
    colours = {
        "NSW": (0,0,128), "VIC": (0,0,180), "QLD": (128,0,0),
        "WA":  (0,100,0), "SA": (180,80,0), "TAS": (60,0,80),
        "ACT": (0,80,80), "NT": (150,75,0),
    }
    hdr_color = colours.get(state, (0,0,128))
    draw.rectangle([0, 0, W, 65], fill=hdr_color)
    draw.text((20, 8),  f"{state} DRIVER LICENCE", fill=(255,255,255),
              font=_font(22, bold=True))
    draw.text((20, 38), "AUSTRALIA", fill=(220,220,220), font=_font(14))

    fields: list[PIIField] = []
    image_id = str(uuid.uuid4())

    face_x, face_y, face_w, face_h = 20, 80, 145, 185
    _draw_face_placeholder(draw, face_x, face_y, face_w, face_h)
    fields.append(PIIField("Face", "face", True,
        BoundingBox(face_x, face_y, face_x+face_w, face_y+face_h).normalize(W, H), "face"))

    tx = 180

    def au_field(label, value, y, font_size=13, pii=True, bold=False):
        draw.text((tx, y), label, fill=(80,80,80), font=_font(9))
        draw.text((tx, y+12), value, fill=(0,0,0), font=_font(font_size, bold=bold))
        vw, vh = _text_size(draw, value, _font(font_size, bold=bold))
        box = BoundingBox(tx, y+12, tx+vw, y+12+vh).normalize(W, H)
        if pii:
            fields.append(PIIField(label, value, True, box, "text"))

    au_field("SURNAME", last, 80, font_size=17, bold=True)
    au_field("GIVEN NAME(S)", first, 114, font_size=14)
    au_field("DATE OF BIRTH", dob, 148)
    au_field("LICENCE NO.", licence, 182, bold=True)
    au_field("EXPIRY", expiry, 212, pii=False)
    au_field("ADDRESS", address, 244)
    au_field("SUBURB / STATE / POSTCODE", f"{suburb} {state} {postcode}", 274)

    sig_x, sig_y, sig_w, sig_h = tx, 310, 210, 45
    draw.text((sig_x, sig_y-12), "SIGNATURE", fill=(80,80,80), font=_font(9))
    draw.rectangle([sig_x, sig_y, sig_x+sig_w, sig_y+sig_h],
                   outline=(160,160,160), width=1)
    _draw_signature_placeholder(draw, sig_x+5, sig_y+3, sig_w-10, sig_h-6,
                                 f"{first} {last}")
    fields.append(PIIField("Signature", f"{first} {last}", True,
        BoundingBox(sig_x, sig_y, sig_x+sig_w, sig_y+sig_h).normalize(W, H), "signature"))

    draw.rectangle([0, 0, W-1, H-1], outline=hdr_color, width=3)

    skew = random.uniform(-5, 5) if random.random() < 0.3 else 0.0
    img = _add_noise_and_effects(img, skew_angle=skew)
    final_w, final_h = img.size

    fname = f"australia_dl_{image_id}.png"
    img_path = output_dir / "images" / fname
    img.save(img_path)

    return CardAnnotation(
        image_id=image_id, country="Australia", card_type="Driver's Licence",
        image_path=str(img_path), image_width=final_w, image_height=final_h,
        fields=fields,
    )


def generate_canada_drivers_licence(output_dir: Path) -> CardAnnotation:
    """Generate a synthetic Canadian driver's licence."""
    fake = FAKERS["Canada"]
    W, H = 856, 540

    first   = fake.first_name()
    last    = fake.last_name().upper()
    dob     = fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%Y/%m/%d")
    expiry  = fake.date_between(start_date="+1y", end_date="+5y").strftime("%Y/%m/%d")
    licence = fake.bothify("?####-#####-#####")
    address = fake.street_address()
    city    = fake.city().upper()
    province = random.choice(["ON", "BC", "AB", "QC", "MB", "SK", "NS", "NB"])
    postal  = fake.postalcode()

    img  = Image.new("RGB", (W, H), (248, 248, 248))
    draw = ImageDraw.ImageDraw(img)

    # Red/white Canadian styling
    draw.rectangle([0, 0, W, 65], fill=(180, 0, 0))
    draw.text((20, 8),  f"PROVINCE OF {province}", fill=(255,255,255),
              font=_font(18, bold=True))
    draw.text((20, 36), "DRIVER'S LICENCE / PERMIS DE CONDUIRE",
              fill=(255,220,220), font=_font(11))

    fields: list[PIIField] = []
    image_id = str(uuid.uuid4())

    face_x, face_y, face_w, face_h = 20, 80, 145, 185
    _draw_face_placeholder(draw, face_x, face_y, face_w, face_h)
    fields.append(PIIField("Face", "face", True,
        BoundingBox(face_x, face_y, face_x+face_w, face_y+face_h).normalize(W, H), "face"))

    tx = 180

    def ca_field(label, value, y, font_size=13, pii=True, bold=False):
        draw.text((tx, y), label, fill=(80,80,80), font=_font(9))
        draw.text((tx, y+12), value, fill=(0,0,0), font=_font(font_size, bold=bold))
        vw, vh = _text_size(draw, value, _font(font_size, bold=bold))
        box = BoundingBox(tx, y+12, tx+vw, y+12+vh).normalize(W, H)
        if pii:
            fields.append(PIIField(label, value, True, box, "text"))

    ca_field("NOM / SURNAME", last, 80, font_size=16, bold=True)
    ca_field("PRÉNOM / GIVEN NAME", first, 112, font_size=14)
    ca_field("NAISSANCE / DATE OF BIRTH", dob, 144)
    ca_field("EXPIRATION", expiry, 172, pii=False)
    ca_field("PERMIS / LICENCE NO.", licence, 200, bold=True)
    ca_field("ADRESSE / ADDRESS", address, 230)
    ca_field("VILLE / CITY, PROVINCE, CODE POSTAL",
             f"{city}, {province}  {postal}", 258)

    sig_x, sig_y, sig_w, sig_h = tx, 295, 210, 45
    draw.rectangle([sig_x, sig_y, sig_x+sig_w, sig_y+sig_h],
                   outline=(160,160,160), width=1)
    _draw_signature_placeholder(draw, sig_x+5, sig_y+3, sig_w-10, sig_h-6,
                                 f"{first} {last}")
    fields.append(PIIField("Signature", f"{first} {last}", True,
        BoundingBox(sig_x, sig_y, sig_x+sig_w, sig_y+sig_h).normalize(W, H), "signature"))

    draw.rectangle([0, 0, W-1, H-1], outline=(180,0,0), width=3)

    skew = random.uniform(-5, 5) if random.random() < 0.3 else 0.0
    img = _add_noise_and_effects(img, skew_angle=skew)
    final_w, final_h = img.size

    fname = f"canada_dl_{image_id}.png"
    img_path = output_dir / "images" / fname
    img.save(img_path)

    return CardAnnotation(
        image_id=image_id, country="Canada", card_type="Driver's Licence",
        image_path=str(img_path), image_width=final_w, image_height=final_h,
        fields=fields,
    )


# ══════════════════════════════════════════════════════════════════════════════
# GENERATOR REGISTRY
# Adding a new country means writing one function and adding it here.
# ══════════════════════════════════════════════════════════════════════════════
GENERATORS = {
    "USA":       generate_usa_drivers_licence,
    "India":     generate_india_aadhaar,
    "UK":        generate_uk_driving_licence,
    "Germany":   generate_germany_personalausweis,
    "Australia": generate_australia_drivers_licence,
    "Canada":    generate_canada_drivers_licence,
}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic ID card images for VLM fine-tuning"
    )
    parser.add_argument("--count", type=int, default=100,
        help="Total number of cards to generate (split evenly across countries)")
    parser.add_argument("--output_dir", type=str, default="./dataset",
        help="Root directory for output images and annotations")
    parser.add_argument("--countries", nargs="+",
        default=list(GENERATORS.keys()),
        help=f"Countries to include. Available: {list(GENERATORS.keys())}")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "annotations").mkdir(parents=True, exist_ok=True)

    # Filter to requested countries
    generators = {k: v for k, v in GENERATORS.items() if k in args.countries}
    if not generators:
        print(f"No valid countries specified. Choose from: {list(GENERATORS.keys())}")
        return

    # Distribute count evenly across countries, rounding up
    per_country = max(1, args.count // len(generators))
    remainder   = args.count - per_country * len(generators)

    manifest = []
    total = 0

    for i, (country, gen_fn) in enumerate(generators.items()):
        n = per_country + (1 if i < remainder else 0)
        print(f"Generating {n} {country} cards ...")

        for j in range(n):
            try:
                annotation = gen_fn(output_dir)
                # Save annotation JSON
                ann_path = output_dir / "annotations" / f"{annotation.image_id}.json"
                ann_path.write_text(json.dumps(annotation.to_dict(), indent=2))
                manifest.append({
                    "image_id":    annotation.image_id,
                    "country":     annotation.country,
                    "card_type":   annotation.card_type,
                    "image_path":  annotation.image_path,
                    "annotation_path": str(ann_path),
                    "pii_count":   len(annotation.fields),
                })
                total += 1
                if (j + 1) % 10 == 0:
                    print(f"  {j+1}/{n} done")
            except Exception as e:
                print(f"  ERROR generating {country} card {j}: {e}")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "total_cards": total,
        "countries":   list(generators.keys()),
        "cards":       manifest,
    }, indent=2))

    print(f"\nDone. Generated {total} cards in {output_dir}/")
    print(f"Country breakdown:")
    from collections import Counter
    counts = Counter(m["country"] for m in manifest)
    for country, count in sorted(counts.items()):
        print(f"  {country}: {count}")
    print(f"\nNext step: run annotate_with_claude.py to generate Claude annotations")
    print(f"  python annotate_with_claude.py --dataset_dir {output_dir}")


if __name__ == "__main__":
    main()
