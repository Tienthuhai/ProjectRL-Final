import pygame
import cv2
import numpy as np
import sys
import os

# --- 1. Xử lý ảnh bằng OpenCV để bóc tách các khối ---
map_path = "maps/map_hard.png"
backup_path = "maps/map_hard_backup.png"

# Ưu tiên lấy từ bản backup để có các khối gốc, nếu không thì lấy map hiện tại
src_img = backup_path if os.path.exists(backup_path) else map_path

img = cv2.imread(src_img)
if img is None:
    print("Không tìm thấy ảnh map!")
    sys.exit()

h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

pixels = img.reshape(-1, 3)
colors, counts = np.unique(pixels, axis=0, return_counts=True)
bg_color_cv = colors[np.argmax(counts)] # BGR
BG_COLOR = (int(bg_color_cv[2]), int(bg_color_cv[1]), int(bg_color_cv[0])) # RGB

# Tạo mask nhận diện vật cản
mask = (gray < 220).astype(np.uint8)

# --- BÓC TÁCH CÁC KHỐI BỊ DÍNH NHAU ---
# Dùng morphological erosion để cắt đứt các điểm dính viền
kernel_erode = np.ones((5, 5), np.uint8)
eroded = cv2.erode(mask, kernel_erode)
num_labels, labels = cv2.connectedComponents(eroded, connectivity=8)

pygame.init()
screen = pygame.display.set_mode((w, h))
pygame.display.set_caption("Kéo thả để thiết kế Map - Nhấn 'S' để lưu")

class DraggableBlock:
    def __init__(self, surface, x, y):
        self.surface = surface
        self.rect = self.surface.get_rect(topleft=(x, y))
        self.dragging = False
        self.offset_x = 0
        self.offset_y = 0

blocks = []
kernel_dilate = np.ones((11, 11), np.uint8)

for i in range(1, num_labels):
    # Lấy label đã được tách, dilate lại để khôi phục viền gốc, sau đó AND với mask ban đầu
    label_mask = (labels == i).astype(np.uint8)
    dilated = cv2.dilate(label_mask, kernel_dilate)
    final_mask = cv2.bitwise_and(dilated, mask)
    
    ys, xs = np.where(final_mask == 1)
    if len(ys) == 0: continue
    
    x, bb_y = np.min(xs), np.min(ys)
    w_bb, h_bb = np.max(xs) - x + 1, np.max(ys) - bb_y + 1
    
    # Cắt BGR
    crop = img[bb_y:bb_y+h_bb, x:x+w_bb].copy()
    
    # Tạo ảnh RGBA để làm trong suốt nền bao quanh vật cản
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    
    # Các pixel không thuộc vật cản này -> cho trong suốt (Alpha = 0)
    crop_mask = final_mask[bb_y:bb_y+h_bb, x:x+w_bb]
    rgba[crop_mask == 0, 3] = 0
    
    # Chuyển BGRA (OpenCV) sang RGBA (Pygame)
    rgba = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)
    
    # Tạo Pygame surface
    surf = pygame.image.frombuffer(rgba.tobytes(), rgba.shape[1::-1], "RGBA")
    blocks.append(DraggableBlock(surf, x, bb_y))

# --- 2. Giao diện Kéo thả bằng Pygame ---
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 22)

# Vị trí an toàn Start/Goal
START_POS = (40, 40)
GOAL_POS = (460, 460)
SAFE_RADIUS = 20

running = True
selected_block = None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_x, mouse_y = event.pos
                # Chọn block từ trên xuống dưới (reverse) để chọn block nằm trên cùng
                for b in reversed(blocks):
                    # Kiểm tra xem có click đúng vào pixel không trong suốt không
                    if b.rect.collidepoint(mouse_x, mouse_y):
                        local_x = mouse_x - b.rect.x
                        local_y = mouse_y - b.rect.y
                        if b.surface.get_at((local_x, local_y)).a > 0:
                            b.dragging = True
                            b.offset_x = b.rect.x - mouse_x
                            b.offset_y = b.rect.y - mouse_y
                            selected_block = b
                            
                            # Đưa block đang kéo lên lớp trên cùng
                            blocks.remove(b)
                            blocks.append(b)
                            break
                        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if selected_block:
                    selected_block.dragging = False
                    selected_block = None
                    
        elif event.type == pygame.MOUSEMOTION:
            if selected_block and selected_block.dragging:
                mouse_x, mouse_y = event.pos
                selected_block.rect.x = mouse_x + selected_block.offset_x
                selected_block.rect.y = mouse_y + selected_block.offset_y
                
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                # --- LƯU MAP ---
                # Chỉ lưu lại Background và các Blocks (Không lưu các vòng tròn báo hiệu Start/Goal)
                save_surf = pygame.Surface((w, h))
                save_surf.fill(BG_COLOR)
                for b in blocks:
                    save_surf.blit(b.surface, b.rect)
                pygame.image.save(save_surf, map_path)
                print(f"Đã lưu map mới vào {map_path} thành công!")

    # 1. Vẽ nền
    screen.fill(BG_COLOR)
    
    # 2. Vẽ hướng dẫn Start/Goal (Vòng tròn an toàn để bạn né ra)
    pygame.draw.circle(screen, (50, 200, 50), START_POS, SAFE_RADIUS, 2)
    pygame.draw.circle(screen, (200, 50, 50), GOAL_POS, SAFE_RADIUS, 2)
    
    txt_start = font.render("START ZONE", True, (50, 180, 50))
    txt_goal = font.render("GOAL ZONE", True, (180, 50, 50))
    screen.blit(txt_start, (START_POS[0] - 40, START_POS[1] + 25))
    screen.blit(txt_goal, (GOAL_POS[0] - 40, GOAL_POS[1] + 25))

    # 3. Vẽ các khối vật cản
    for b in blocks:
        screen.blit(b.surface, b.rect)
        
    # 4. In hướng dẫn
    instructions = font.render("Dung chuot de keo tha. Nhan 'S' de luu vao map_hard.png", True, (0,0,0))
    screen.blit(instructions, (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
