import taichi as ti
import open3d as o3d
import numpy as np


def get_spring_mass_from_pcd(pcd, raidus=0.05, max_neighbours=11):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    spring_flags = np.zeros((len(points), len(points)))
    springs = []
    rest_lengths = []
    vertices = points  # Use the points as the vertices of the springs
    for i in range(len(vertices)):
        [k, idx, _] = pcd_tree.search_hybrid_vector_3d(points[i], raidus, max_neighbours)
        idx = idx[1:]
        for j in idx:
            if spring_flags[i, j] == 0 and spring_flags[j, i] == 0:
                spring_flags[i, j] = 1
                spring_flags[j, i] = 1
                springs.append([i, j])
                rest_lengths.append(np.linalg.norm(points[i] - points[j]))
    return vertices, springs, rest_lengths


def get_spring_mass_visual(vertices, springs, rest_legnths):
    # Color the springs with force information
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(vertices)
    lineset.lines = o3d.utility.Vector2iVector(springs)
    line_colors = [
        np.array([0.0, 1.0, 0.0])
        * np.abs(
            (
                1
                - np.linalg.norm(vertices[springs[i][0]] - vertices[springs[i][1]])
                / rest_legnths[i]
            )
        )
        for i in range(len(springs))
    ]
    lineset.colors = o3d.utility.Vector3dVector(line_colors)

    # Color the vertices
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.paint_uniform_color([1, 0, 0])

    visuals = [lineset, pcd]
    return visuals


def demo1():
    # Load the table into taichi and create a simple spring-mass system
    substep = 10
    table = o3d.io.read_point_cloud("data/table.ply")
    # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    # o3d.visualization.draw_geometries([table, coordinate])
    vertices, springs, rest_lengths = get_spring_mass_from_pcd(table)
    # visuals = get_spring_mass_visual(vertices, springs, rest_lengths)
    # o3d.visualization.draw_geometries(visuals)

    # Setup the physics part
    x = ti.Vector.field(3, dtype=ti.f32, shape=len(vertices))
    x.from_numpy(vertices)
    v = ti.Vector.field(3, dtype=ti.f32, shape=len(vertices))
    
    # Spring-mass system


    


    

    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    ti.init(arch=ti.gpu)

    demo1()
