import json
import sys
from math import cos, radians, sin

import numpy as np
from PIL import Image


class Util:
    @staticmethod
    def normalize(x: np.ndarray):
        return x / np.linalg.norm(x)


class Camera:
    def __init__(self, camera_json: object):

        # Set bounds
        self.bounds = Bounds(camera_json["bounds"])

        # Set R, U, N, V values
        camera_from_json = camera_json["from"]
        camera_to_json = camera_json["to"]

        self.camera_from = np.array(camera_from_json)
        self.camera_to = np.array(camera_to_json)

        # R = Camera "location"
        self.r = self.camera_from - self.camera_to

        # N = Camera Z-Axis
        # Initially assumed to be = Normalized R
        self.n = Util.normalize(self.r)

        # U = Camera X-Axis
        # We are making the assumption that the camera's Y-axis is [0,1,0] to find X
        # Then to find the actual Y-Axis, we cross this X-axis value with the Z-axis value
        self.u = np.cross([0, 1, 0], self.n)

        # V = Camera Y-Axis
        # By finding the X-Axis above, can calculate the actual Y-axis
        self.v = np.cross(self.n, self.u)

        self.u = Util.normalize(self.u)
        self.v = Util.normalize(self.v)

        # Compute the View and Perspective Projection Matrices
        self.__compute_view_matrix()
        self.__compute_perspective_projection_matrix()
        self.__compute_view_perspective_projection_composite_matrix()

    def __compute_view_matrix(self):
        self.view_matrix = np.array(
            [
                [
                    self.u[0],
                    self.u[1],
                    self.u[2],
                    -1 * np.dot(self.r, self.u),
                ],
                [
                    self.v[0],
                    self.v[1],
                    self.v[2],
                    -1 * np.dot(self.r, self.v),
                ],
                [
                    self.n[0],
                    self.n[1],
                    self.n[2],
                    -1 * np.dot(self.r, self.n),
                ],
                [0, 0, 0, 1],
            ]
        )

    def __compute_perspective_projection_matrix(self):
        self.perspective_projection_matrix = np.array(
            [
                [
                    2 * self.bounds.near / (self.bounds.right - self.bounds.left),
                    0,
                    (self.bounds.right + self.bounds.left)
                    / (self.bounds.right - self.bounds.left),
                    0,
                ],
                [
                    0,
                    (2 * self.bounds.near) / (self.bounds.top - self.bounds.bottom),
                    (self.bounds.top + self.bounds.bottom)
                    / (self.bounds.top - self.bounds.bottom),
                    0,
                ],
                [
                    0,
                    0,
                    -1
                    * (self.bounds.far + self.bounds.near)
                    / (self.bounds.far - self.bounds.near),
                    -1
                    * (2 * self.bounds.far * self.bounds.near)
                    / (self.bounds.far - self.bounds.near),
                ],
                [0, 0, -1, 0],
            ]
        )

    def __compute_view_perspective_projection_composite_matrix(self):
        self.view_perspective_projection_composite_matrix = np.dot(
            self.perspective_projection_matrix, self.view_matrix
        )


class Bounds:
    def __init__(self, bounds_json: object):
        self.near = bounds_json[0]
        self.far = bounds_json[1]
        self.right = bounds_json[2]
        self.left = bounds_json[3]
        self.top = bounds_json[4]
        self.bottom = bounds_json[5]


class Material:
    def __init__(self, material_json: object):
        self.Cs = material_json["Cs"]
        self.Ka = material_json["Ka"]
        self.Kd = material_json["Kd"]
        self.Ks = material_json["Ks"]
        self.n = material_json["n"]


class Transforms:
    def __init__(self, transforms_json):
        self.transforms = transforms_json
        self.__compute_transform_matrix()

    def __compute_transform_matrix(self):

        self.transforms_matrix = np.identity(4)

        for transform in self.transforms:
            (transform_key, transform_value) = next(iter(transform.items()))

            if transform_key == "Ry":
                self.transforms_matrix = np.dot(
                    [
                        [
                            cos(radians(transform_value)),
                            0,
                            sin(radians(transform_value)),
                            0,
                        ],
                        [0, 1, 0, 0],
                        [
                            -1 * sin(radians(transform_value)),
                            0,
                            cos(radians(transform_value)),
                            0,
                        ],
                        [0, 0, 0, 1],
                    ],
                    self.transforms_matrix,
                )

            elif transform_key == "S":
                self.transforms_matrix = np.dot(
                    [
                        [transform_value[0], 0, 0, 0],
                        [0, transform_value[1], 0, 0],
                        [0, 0, transform_value[2], 0],
                        [0, 0, 0, 1],
                    ],
                    self.transforms_matrix,
                )
            elif transform_key == "T":
                self.transforms_matrix = np.dot(
                    [
                        [1, 0, 0, transform_value[0]],
                        [0, 1, 0, transform_value[1]],
                        [0, 0, 1, transform_value[2]],
                        [0, 0, 0, 1],
                    ],
                    self.transforms_matrix,
                )


class Vertex:
    def __init__(self, vertex_json):
        # Vertex coordinates
        # Append the homogeneous coordinate
        self.v = np.concatenate((vertex_json["v"], [1]), dtype=np.float64)

        # Vertex normals
        self.n = np.array(vertex_json["n"], dtype=np.float64)

        # Texture coordinate
        self.t = np.array(vertex_json["t"], dtype=np.float64)


class Triangle:
    def __init__(self, triangle_json: object):
        self.vertices: list[Vertex] = []
        for _, vertex_json in triangle_json.items():
            self.vertices.append(Vertex(vertex_json))

        # self.color = (255, 255, 255)
        self.compute_color()

    def x_min(self):
        return min(self.vertices, key=lambda vertex: vertex.v[0]).v[0]

    def x_max(self):
        return max(self.vertices, key=lambda vertex: vertex.v[0]).v[0]

    def y_min(self):
        return min(self.vertices, key=lambda vertex: vertex.v[1]).v[1]

    def y_max(self):
        return max(self.vertices, key=lambda vertex: vertex.v[1]).v[1]

    def compute_color(self):
        dot_p = np.dot([0.707, 0.5, 0.5], self.vertices[0].n)

        if dot_p < 0:
            dot_p *= -1
        elif dot_p > 1.0:
            dot_p = 1.0

        self.color = (
            int(0.95 * dot_p * 255),
            int(0.65 * dot_p * 255),
            int(0.88 * dot_p * 255),
        )

        self.color = tuple((255 * np.dot([0.95, 0.65, 0.88], dot_p)).astype(int))

    def f01(self, x: np.float64, y: np.float64):
        # f01(x,y) = (y0-y1)x + (x1-x0)y + x0y1-x1y0
        return (
            (self.vertices[0].v[1] - self.vertices[1].v[1]) * x
            + (self.vertices[1].v[0] - self.vertices[0].v[0]) * y
            + (self.vertices[0].v[0] * self.vertices[1].v[1])
            - (self.vertices[1].v[0] * self.vertices[0].v[1])
        )

    def f12(self, x: np.float64, y: np.float64):
        # f12(x,y) = (y1-y2)x + (x2-x1)y + x1y2-x2y1
        return (
            (self.vertices[1].v[1] - self.vertices[2].v[1]) * x
            + (self.vertices[2].v[0] - self.vertices[1].v[0]) * y
            + (self.vertices[1].v[0] * self.vertices[2].v[1])
            - (self.vertices[2].v[0] * self.vertices[1].v[1])
        )

    def f20(self, x: np.float64, y: np.float64):
        # f20(x,y) = (y2-y0)x + (x0-x2)y + x2y0-x0y2
        return (
            (self.vertices[2].v[1] - self.vertices[0].v[1]) * x
            + (self.vertices[0].v[0] - self.vertices[2].v[0]) * y
            + (self.vertices[2].v[0] * self.vertices[0].v[1])
            - (self.vertices[0].v[0] * self.vertices[2].v[1])
        )


class Shape:
    def __init__(self, shape_json: object):
        self.id = shape_json["id"]
        self.notes = shape_json["notes"]
        self.geometry = shape_json["geometry"]
        self.material = Material(shape_json["material"])
        self.transforms = Transforms(shape_json["transforms"])

        # Note: Load json files independently so avoid reference sharing
        shape_file = open(f"{self.geometry}.json")
        geometry_json = json.load(shape_file)

        self.triangles: list[Triangle] = []

        for triangle_json in geometry_json["data"]:
            self.triangles.append(Triangle(triangle_json))

        print(
            f"Loaded '{self.id}' from '{self.geometry}.json'. Notes: {self.notes if self.notes else '(none)'}."
        )


class Canvas:
    def __init__(self, scene_file_name: str):
        print(f"Scene file: '{scene_file_name}':")

        scene_file = open(scene_file_name)
        scene_json = json.load(scene_file)["scene"]

        shapes_json = scene_json["shapes"]
        lights_json = scene_json["lights"]
        camera_json = scene_json["camera"]
        resolution_json = camera_json["resolution"]

        # Set camera
        self.camera = Camera(camera_json)
        # Set bounds

        self.width: int = resolution_json[0]
        self.height: int = resolution_json[1]

        # Initialize empty shapes list
        self.shapes: list[Shape] = []
        for shape_json in shapes_json:
            self.shapes.append(Shape(shape_json))

        # Create Pillow image
        self.im = Image.new("RGB", (self.width, self.height))

        # Initialize Z-Buffer
        self.z_buffer: list[list[float]] = [
            [sys.float_info.max for _ in range(self.width)] for _ in range(self.height)
        ]

        # Draw the models at the end
        self.draw_models()

    def show(self):
        self.im.show()

    def draw_models(self):
        # Transform Vertices
        for shape in self.shapes:
            for triangle in shape.triangles:
                for vertex in triangle.vertices:

                    # # Transform the shape in Object space
                    vertex.v = np.dot(shape.transforms.transforms_matrix, vertex.v)

                    # Transform Vertex from World Space to NDC Space
                    vertex.v = np.dot(
                        self.camera.view_perspective_projection_composite_matrix,
                        vertex.v,
                    )
                    vertex.v /= vertex.v[3]

                    # Normalize Vertex to Screen Space
                    vertex.v[0] = ((vertex.v[0] + 1) * (self.width - 1)) / 2
                    vertex.v[1] = ((1 - vertex.v[1]) * (self.height - 1)) / 2

                self.draw_triangle(triangle)

    def draw_triangle(self, triangle: Triangle):
        x_min = int(triangle.x_min())
        x_max = int(triangle.x_max())
        y_min = int(triangle.y_min())
        y_max = int(triangle.y_max())

        for y in range(max(0, y_min), min(self.height, y_max + 1)):
            for x in range(max(0, x_min), min(self.width, x_max + 1)):
                alpha = triangle.f12(x, y) / triangle.f12(
                    triangle.vertices[0].v[0], triangle.vertices[0].v[1]
                )
                beta = triangle.f20(x, y) / triangle.f20(
                    triangle.vertices[1].v[0], triangle.vertices[1].v[1]
                )
                gamma = triangle.f01(x, y) / triangle.f01(
                    triangle.vertices[2].v[0], triangle.vertices[2].v[1]
                )

                if alpha >= 0 and beta >= 0 and gamma >= 0:
                    z = (
                        alpha * triangle.vertices[0].v[2]
                        + beta * triangle.vertices[1].v[2]
                        + gamma * triangle.vertices[2].v[2]
                    )

                    if z < self.z_buffer[y][x]:
                        self.im.putpixel((x, y), triangle.color)
                        self.z_buffer[y][x] = z


def main():
    canvas = Canvas("scene.json")
    canvas.show()


if __name__ == "__main__":
    main()
