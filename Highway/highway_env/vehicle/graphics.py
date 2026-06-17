from __future__ import annotations
import numpy as np
import pygame
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle

class VehicleGraphics:
    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLUE = (100, 200, 255)
    YELLOW = (200, 200, 0)
    BLACK = (60, 60, 60)
    DEFAULT_COLOR = YELLOW
    EGO_COLOR = GREEN

    @classmethod
    def display(cls, vehicle, surface, transparent=False, offscreen=False, label=False):
        if not surface.is_visible(vehicle.position): return

        v = vehicle
        length = v.LENGTH
        vehicle_surface = pygame.Surface((surface.pix(length), surface.pix(length)), flags=pygame.SRCALPHA)
        
        # Draw Body
        rect = (surface.pix(length/2 - v.LENGTH/2), surface.pix(length/2 - v.WIDTH/2), surface.pix(v.LENGTH), surface.pix(v.WIDTH))
        color = cls.get_color(v, transparent)
        pygame.draw.rect(vehicle_surface, color, rect, 0)
        pygame.draw.rect(vehicle_surface, cls.BLACK, rect, 1)

        # Rotate and Blit
        pos = surface.pos2pix(v.position[0], v.position[1])
        cls.blit_rotate(surface, vehicle_surface, pos, np.rad2deg(-v.heading))

    @staticmethod
    def blit_rotate(surf, image, pos, angle):
        w, h = image.get_size()
        box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
        box_rotate = [p.rotate(angle) for p in box]
        min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
        max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])
        pivot = pygame.math.Vector2(w / 2, -h / 2)
        pivot_rotate = pivot.rotate(angle)
        pivot_move = pivot_rotate - pivot
        origin = (pos[0] - w / 2 + min_box[0] - pivot_move[0], pos[1] - h / 2 - max_box[1] + pivot_move[1])
        rotated_image = pygame.transform.rotate(image, angle)
        surf.blit(rotated_image, origin)

    @classmethod
    def get_color(cls, vehicle, transparent=False):
        color = cls.DEFAULT_COLOR
        if vehicle.crashed: color = cls.RED
        elif isinstance(vehicle, MDPVehicle): color = cls.EGO_COLOR
        elif isinstance(vehicle, IDMVehicle): color = cls.BLUE
        
        if transparent: color = (color[0], color[1], color[2], 50)
        return color