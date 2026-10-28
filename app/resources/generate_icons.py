"""
Generate application icons for Sign Language Recognition Studio.
Creates high-res PNG and Windows multi-resolution ICO file.
"""
from PIL import Image, ImageDraw

def create_app_icon():
    size = 512
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background rounded squircle
    radius = 110
    draw.rounded_rectangle(
        [(16, 16), (size - 16, size - 16)],
        radius=radius,
        fill=(15, 17, 23, 255),
        outline=(94, 106, 210, 255),
        width=12
    )

    # Inner subtle glow ring
    draw.rounded_rectangle(
        [(36, 36), (size - 36, size - 36)],
        radius=radius - 16,
        outline=(35, 42, 60, 255),
        width=4
    )

    # Hand gesture / motion vector geometric motif
    # Palm base
    palm_center = (256, 340)
    draw.rounded_rectangle(
        [(180, 280), (332, 410)],
        radius=40,
        fill=(94, 106, 210, 255)
    )

    # 4 fingers (upright & dynamic)
    fingers = [
        (190, 160, 216, 290), # Index
        (230, 130, 256, 290), # Middle (tallest)
        (270, 150, 296, 290), # Ring
        (310, 190, 336, 290), # Pinky
    ]
    for x1, y1, x2, y2 in fingers:
        draw.rounded_rectangle([(x1, y1), (x2, y2)], radius=14, fill=(241, 245, 249, 255))

    # Thumb angled outward
    draw.polygon([(180, 310), (130, 260), (150, 240), (200, 290)], fill=(241, 245, 249, 255))
    draw.ellipse([(120, 235), (155, 270)], fill=(241, 245, 249, 255))

    # Spatial Landmark Nodes (signifying computer vision / MediaPipe joints)
    joint_nodes = [
        (256, 345), # Wrist/palm center
        (203, 165), # Index tip
        (243, 135), # Middle tip
        (283, 155), # Ring tip
        (323, 195), # Pinky tip
        (135, 250), # Thumb tip
    ]
    for x, y in joint_nodes:
        draw.ellipse([(x - 8, y - 8), (x + 8, y + 8)], fill=(56, 189, 248, 255), outline=(255, 255, 255, 255), width=2)

    # Dynamic trajectory arc (signifying dynamic J / Z signs)
    draw.arc([(80, 100), (432, 452)], start=210, end=330, fill=(56, 189, 248, 220), width=6)

    # Save PNG
    png_path = "app/resources/app_icon.png"
    img.save(png_path, "PNG")

    # Save ICO with multiple sizes for crisp Windows rendering
    ico_path = "app/resources/app_icon.ico"
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    img.save(ico_path, format="ICO", sizes=sizes)
    print(f"Icons generated: {png_path}, {ico_path}")

if __name__ == "__main__":
    create_app_icon()
