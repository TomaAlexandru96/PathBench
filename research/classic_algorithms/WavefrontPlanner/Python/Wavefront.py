# Wavefront Planning
# Sajad Saeedi, Andrew Davison 2017
# Implementation is based on the following reference
#
# H. Choset, K. M. Lynch, S. Hutchinson, G. Kantor, W. Burgard, L. E. Kavraki and S. Thrun,
# Principles of Robot Motion: Theory, Algorithms, and Implementations,
# MIT Press, Boston, 2005.
# http://www.cs.cmu.edu/afs/cs/Web/People/motionplanning/ 
#
# From Chapter 4.5 - Wave-Front Planner
# This planner determines a path via gradient descent on the grid starting from the start. 
# Essentially, the planner determines the path one pixel at a time.
# The wave-front planner essentially forms a potential function on the grid which has one local minimum and thus is resolution complete. 
# The planner also determines the shortest path, but at the cost of coming dangerously close to obstacles. 
# The major drawback of this method is that the planner has to search the entire space for a path

import pygame, os, math, time, random
import numpy as np
try:
    import pygame
    from pygame import surfarray
    from pygame.locals import *
except ImportError:
    raise ImportError('Error Importing Pygame/surfarray')
    
pygame.init()

SCALE = 2

# set the width and height of the screen
WIDTH = 1500/SCALE
HEIGHT = 1000/SCALE

# The region we will fill with obstacles
PLAYFIELDCORNERS = (-3.0, -3.0, 3.0, 3.0)

# Barrier locations
barriers = []
# barrier contents are (bx, by, visibilitymask)
for i in range(100):
	(bx, by) = (random.uniform(PLAYFIELDCORNERS[0], PLAYFIELDCORNERS[2]), random.uniform(PLAYFIELDCORNERS[1], PLAYFIELDCORNERS[3]))
	barrier = [bx, by, 0]
	barriers.append(barrier)

BARRIERRADIUS = 0.1


ROBOTRADIUS = 0.15
W = 2 * ROBOTRADIUS
SAFEDIST = 0.2
BARRIERINFILATE = ROBOTRADIUS

MAXVELOCITY = 0.5     #ms^(-1) max speed of each wheel
MAXACCELERATION = 0.5 #ms^(-2) max rate we can change speed of each wheel

target = (PLAYFIELDCORNERS[2] + 1.0, 0)

k = 160/SCALE # pixels per metre for graphics
u0 = WIDTH / 2
v0 = HEIGHT / 2

x = PLAYFIELDCORNERS[0] - 0.5
y = 0.0
theta = 0.0

vL = 0.00
vR = 0.00

size = [WIDTH, HEIGHT]
screen = pygame.display.set_mode(size)
black = (20,20,40)

# This makes the normal mouse pointer invisible in graphics window
pygame.mouse.set_visible(0)

# time delta
dt = 0.1


def surfdemo_show(array_img, name):
	"displays a surface, waits for user to continue"
	screen = pygame.display.set_mode(array_img.shape[:2], 0, 32)
	surfarray.blit_array(screen, array_img)
	pygame.display.flip()
	pygame.display.set_caption(name)
	while 1:
		e = pygame.event.wait()
		if e.type == MOUSEBUTTONDOWN: break
		if e.type == KEYDOWN: break


def predictPosition(vL, vR, x, y, theta, dt):

	# Position update in time dt

	# Special cases
	# Straight line motion
	if (vL == vR): 
		xnew = x + vL * dt * math.cos(theta)
		ynew = y + vL * dt * math.sin(theta)
		thetanew = theta
	# Pure rotation motion
	elif (vL == -vR):
		xnew = x
		ynew = y
		thetanew = theta + ((vR - vL) * dt / W)
	else:
		# Rotation and arc angle of general circular motion
		R = W / 2.0 * (vR + vL) / (vR - vL)
		deltatheta = (vR - vL) * dt / W
		xnew = x + R * (math.sin(deltatheta + theta) - math.sin(theta))
		ynew = y - R * (math.cos(deltatheta + theta) - math.cos(theta))
		thetanew = theta + deltatheta
	
	return (xnew, ynew, thetanew)


def drawBarriers(barriers):
	for barrier in barriers:
		if(barrier[2] == 0):
			pygame.draw.circle(screen, (0,20,80), (int(u0 + k * barrier[0]), int(v0 - k * barrier[1])), int(k * BARRIERRADIUS), 0)
		else:
			pygame.draw.circle(screen, (0,120,255), (int(u0 + k * barrier[0]), int(v0 - k * barrier[1])), int(k * BARRIERRADIUS), 0)
	return
	
def dialtebarrieres(imgin, barriers):
	imgout = imgin
	for barrier in barriers:
		if(barrier[2] == 0):
			center_u = int(u0 + k * barrier[0])
			center_v = int(v0 - k * barrier[1])
			RAD = BARRIERRADIUS+BARRIERINFILATE
			radius = int(k * (RAD))
			radius2 = int(k * (BARRIERRADIUS))
			points = [(x,y) for x in range(center_u-radius, center_u+radius) for y in range(center_v-radius, center_v+radius)]
			for pt in points:
				vu = center_u - pt[0]
				vv = center_v - pt[1]
				distance = math.sqrt(vv*vv + vu*vu)
				if (distance < radius and distance > radius2 and pt[0] < WIDTH-1 and pt[1] < HEIGHT-1):
					imgout[pt[0],pt[1],0] = 0
					imgout[pt[0],pt[1],1] = 40
					imgout[pt[0],pt[1],2] = 80
	return imgout
	
def MakeWaveFront(imgarray, start_uv, target_uv):
	print "building wavefront, please wait ... "	
	heap = []
	newheap = []

	u = target_uv[0]
	v = target_uv[1]

	imgarray[:,:,0] = 0 # use channel 0 for planning
	imgarray[u, v, 0] = 2

	lastwave = 3 	# start by setting nodes around target to 3
	moves = [(u + 1, v), (u - 1, v), (u, v - 1), (u, v + 1)]
	for move in moves:
		imgarray[move[0], move[1], 0] = 3
		heap.append(move)

	path = False
	max_search = int(np.sqrt(WIDTH*WIDTH + HEIGHT*HEIGHT))
	for currentwave in range(4, max_search):
		lastwave = lastwave + 1
		while(heap != []):
			position = heap.pop()
			(u, v) = position
			moves = [(u + 1, v), (u - 1, v), (u, v + 1), (u, v - 1)]
			for move in moves:
				if(imgarray[move[0], move[1], 2] != 80):
					if(imgarray[move[0], move[1], 0] == 0 and imgarray[position[0], position[1], 0] == currentwave - 1 and move[0] < WIDTH-1 and move[1] < HEIGHT-1 and move[0]>2 and move[1]>2):
						imgarray[move[0], move[1], 0] = currentwave
						newheap.append(move)
					if(move == start_uv):
						path = True
						break 	
			if(path == True):
				break			
		if(path == True):
			break
		heap = newheap
		newheap = []
	return imgarray, currentwave


def FindPath(imgarray, start_uv, currentwave):
	trajectory = []	
	nextpt = start_uv
	path = []
	for backwave in range(currentwave-1,2,-1):
		path.append(nextpt)
		(u,v) = nextpt
		values = []
		val = []
		moves = [(u + 1, v), (u - 1, v), (u, v + 1), (u, v - 1), (u + 1, v + 1), (u - 1, v - 1), (u + 1, v - 1), (u - 1, v - 1)]
		for move in moves:
			val.append(imgarray[move[0], move[1], 0])
			if(imgarray[move[0], move[1], 0] == backwave):
				values.append(imgarray[move[0], move[1], 0])
		minimum = min(values)
		indices = [i for i, v in enumerate(val) if v == minimum]
		nextid = random.choice(indices)
		nextpt = moves[nextid]	
	return path

def GetControl(x,y,theta, waypoint):
	wpu = waypoint[0]
	wpv = waypoint[1]
	
	vt = 0.2
	u = u0 + k * x
	v = v0 - k * y
	vector = (wpu - u, -wpv + v)
	vectorangle = math.atan2(vector[1], vector[0])
	psi = vectorangle - theta

	if(abs(psi) > (2*3.14/180)):
		if(psi > 0):
			vR = vt/2
			vL = -vt/2
		else:
			vR = -vt/2
			vL = vt/2
	else:
		wlx = x - (W/2.0) * math.sin(theta)
		wly = y + (W/2.0) * math.cos(theta)	
		ulx = u0 + k * wlx
		vlx = v0 - k * wly
		dl = math.sqrt((ulx-wpu)*(ulx-wpu) + (vlx-wpv)*(vlx-wpv))  
		
		wrx = x + (W/2.0) * math.sin(theta)
		wry = y - (W/2.0) * math.cos(theta)
		urx = u0 + k * wrx
		vrx = v0 - k * wry
		dr = math.sqrt((urx-wpu)*(urx-wpu) + (vrx-wpv)*(vrx-wpv))
		
		vR = 1*(2*vt)/(1+(dl/dr))
		vL = 1*(2*vt - vR)

	return vL, vR

def AnimateRobot(x,y,theta):
	# Draw robot
	u = u0 + k * x
	v = v0 - k * y
	pygame.draw.circle(screen, (255,255,255), (int(u), int(v)), int(k * ROBOTRADIUS), 3)
	# Draw wheels
	# left wheel centre 
	wlx = x - (W/2.0) * math.sin(theta)
	wly = y + (W/2.0) * math.cos(theta)
	ulx = u0 + k * wlx
	vlx = v0 - k * wly
	WHEELBLOB = 0.04
	pygame.draw.circle(screen, (0,0,255), (int(ulx), int(vlx)), int(k * WHEELBLOB))
	# right wheel centre 
	wrx = x + (W/2.0) * math.sin(theta)
	wry = y - (W/2.0) * math.cos(theta)
	urx = u0 + k * wrx
	vrx = v0 - k * wry
	pygame.draw.circle(screen, (0,0,255), (int(urx), int(vrx)), int(k * WHEELBLOB))

	time.sleep(dt / 5)
	pygame.display.flip()
	#time.sleep(0.2)

def AnimateNavigation(barriers, waypint, path):
	screen.fill(black)
	drawBarriers(barriers)
	
	for pt in path:
		screen.set_at(pt, (255,255,255))
	
	pygame.draw.circle(screen, (255,100,0), target_uv, int(k * ROBOTRADIUS), 0)		
	pygame.draw.circle(screen, (255,100,0), (int(u0 + k * target[0]), int(v0 - k * target[1])), int(k * ROBOTRADIUS), 0)
	pygame.draw.circle(screen, (255,0,255), (int(waypint[0]), int(waypint[1])), int(5))
		
def AnimatePath(imgarray, path, start_uv):
	for pt in path:
		imgarray[pt[0], pt[1], 0] = 0
		imgarray[pt[0], pt[1], 1] = 255
		imgarray[pt[0], pt[1], 2] = 255
				
	#imgarray[:,:,0] /= imgarray[:,:,0].max()/255.0
	scalefactor = imgarray[:,:,0].max()/255.0
	imgarray[:,:,0] = imgarray[:,:,0]/scalefactor
	imgarray[:,:,0].astype(int)
	imgarray[start_uv[0], start_uv[1], 0] = 0
	imgarray[start_uv[0], start_uv[1], 1] = 255
	imgarray[start_uv[0], start_uv[1], 2] = 255

	surfdemo_show(imgarray, 'Wavefront Path Planning')	


def GetWaypoint(x,y,theta, path, waypointIndex, waypointSeperation, target):
	reset = False
	waypoint = path[waypointIndex]
	u = u0 + k * x
	v = v0 - k * y
	distance_to_wp = math.sqrt((waypoint[0]-u)**2+(waypoint[1]-v)**2) # todo compare distance in metrics
	if(distance_to_wp < 3):
		waypointIndex = waypointSeperation + waypointIndex 
		if(waypointIndex > len(path)):
			waypointIndex = len(path) - 1

	return waypoint, reset, waypointIndex

def IsAtTarget(x,y,target):
	disttotarget = math.sqrt((x - target[0])**2 + (y - target[1])**2)
	if (disttotarget < 0.04):
		return True
	else:
		return False


while(1):
	start_uv  = (int(u0 + k * x), int(v0 - k * y)) 
	target_uv = (int(u0 + k * target[0]), int(v0 - k * target[1]))
	print "start is: ", start_uv
	print "goal  is: ", target_uv

	# prepare map of the world for plannign
	screen.fill(black)
	drawBarriers(barriers)
	pygame.draw.circle(screen, (255,100,0), target_uv, int(k * ROBOTRADIUS), 0)
	pygame.draw.circle(screen, (255,255,0), start_uv,  int(k * ROBOTRADIUS), 0)
	imgscreen8  = pygame.surfarray.array3d(screen)
	imgscreen16 = np.array(imgscreen8, dtype=np.uint16)
	drawBarriers(barriers)
	imgarray = dialtebarrieres(imgscreen16, barriers)
	pygame.display.flip()
	pygame.display.set_caption('Wavefront Path Planning')
	
	# build wavefront, given the map, start point, and target point
	imgarray, currentwave = MakeWaveFront(imgarray, start_uv, target_uv)
	print "press a key to start navaigation ... "

	# find the path using the wavefront	
	path = FindPath(imgarray, start_uv, currentwave)
	
	# normlaize and show the path on the map
	AnimatePath(imgarray, path, start_uv)

	# set start point
	x = PLAYFIELDCORNERS[0] - 0.5
	y = 0
	theta = 0
	waypointSeperation = 40;
	waypointIndex = waypointSeperation;

	# loop to navigate the path from start to target	
	while(1):		
		# get a waypoint to follow the path
		(waypoint, reset, waypointIndex)= GetWaypoint(x,y,theta, path, waypointIndex, waypointSeperation, target)

		# reset the simulation, if robot is at target
		if (IsAtTarget(x,y,target)): 
			target = (target[0], random.uniform(PLAYFIELDCORNERS[1], PLAYFIELDCORNERS[3]))
			x = PLAYFIELDCORNERS[0] - 0.5
			y = 0.0
			theta = 0.0
			break


		# calculate control signals to get to the next waypoint
		(vL, vR) = GetControl(x,y,theta, waypoint)

		# Actually now move and update position based on chosen vL and vR
		(x, y, theta) = predictPosition(vL, vR, x, y, theta, dt)

		# animate 	
		AnimateNavigation(barriers, waypoint, path)
		AnimateRobot(x,y,theta)	
		
