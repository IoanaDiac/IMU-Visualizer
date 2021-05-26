import sys, math, pygame, serial
from operator import itemgetter
import pandas as pd
from complementary import df_ang
from kalman import df_kalman

df_test=df_kalman

class Point3D:
    def __init__(self, x=0, y=0, z=0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def rotate_X(self, alfa):
        rad = alfa * math.pi / 180
        cosa = math.cos(rad)
        sina = math.sin(rad)
        y = self.y * cosa - self.z * sina
        z = self.y * sina + self.z * cosa
        return Point3D(self.x, y, z)

    def rotate_Y(self, alfa):
        rad = alfa * math.pi / 180
        cosa = math.cos(rad)
        sina = math.sin(rad)
        z = self.z * cosa - self.x * sina
        x = self.z * sina + self.x * cosa
        y = self.y
        return Point3D( x, y, z)

    def rotate_Z(self, alfa):
        rad = alfa * math.pi / 180
        cosa = math.cos(rad)
        sina = math.sin(rad)
        x = self.x * cosa - self.y * sina
        y = self.x * sina + self.y * cosa
        z = self.z
        return Point3D(x, y, z)

    def project(self, width, height, fov, viewer_dist):
        factor = fov / (viewer_dist + self.z)
        x = self.x * factor + width / 2
        y = self.y * factor + height / 2
        return Point3D(x, y, 1)


class Simulation:
    def __init__(self, width = 640, height = 480):
        pygame.init()

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("3D Simulation")

        self.clock = pygame.time.Clock()

        self.vertices = [
            Point3D(-1, 1, -1),
            Point3D(1, 1, -1),
            Point3D(1, -1, -1),
            Point3D(-1, -1, -1),
            Point3D(-1, 1, 1),
            Point3D(1, 1, 1),
            Point3D(1, -1, 1),
            Point3D(-1, -1, 1)
        ]

        self.faces = [(0, 1, 2, 3), (1, 5, 6, 2), (5, 4, 7, 6), (4, 0, 3, 7), (0, 4, 5, 1), (3, 2, 6, 7)]

        self.colors = [(222, 123, 179), (255, 0, 0), (18, 204, 171), (0, 0, 255), (0, 255, 255), (255, 255, 0)]

        self.angleX, self.angleY, self.angleZ = 0, 0, 0

    def run(self):
        count=0
        while 1 and count<1999:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
            self.clock.tick(200)
            self.screen.fill((0, 0, 0))

            t = []

            for v in self.vertices:
                r = v.rotate_X(self.angleX).rotate_Y(self.angleY).rotate_Z(self.angleZ)
                p = r.project(self.screen.get_width(), self.screen.get_height(), 256, 4)
                t.append(p)

            avg_z = []
            i = 0
            for f in self.faces:
                z = (t[f[0]].z + t[f[1]].z + t[f[2]].z + t[f[3]].z) / 4.0
                avg_z.append([i, z])
                i = i + 1

            for tmp in sorted(avg_z, key=itemgetter(1), reverse=True):
                face_index = tmp[0]
                f = self.faces[face_index]
                pointlist = [(t[f[0]].x, t[f[0]].y), (t[f[1]].x, t[f[1]].y),
                             (t[f[1]].x, t[f[1]].y), (t[f[2]].x, t[f[2]].y),
                             (t[f[2]].x, t[f[2]].y), (t[f[3]].x, t[f[3]].y),
                             (t[f[3]].x, t[f[3]].y), (t[f[0]].x, t[f[0]].y)]
                pygame.draw.polygon(self.screen, self.colors[face_index], pointlist)


            self.angleX=df_test.at[count,'rotx']
            self.angleY=df_test.at[count,'roty']
            self.angleZ=df_test.at[count,'rotz']
            count+=1
            pygame.display.flip()



if __name__=="__main__":
    Simulation().run()
    # ser.close()
