
import numpy as np
import pygame
import random
import carla
import logging

def xy2lms(points):
    """
    points : (n,2) or (n,3), vector of (x,y,[z])
    
    
    """
    yaw = np.arctan2(points[:,1],points[:,0])#the -y axis is the start of rotation
    d = np.sqrt(points[:,0]**2 + points[:,1]**2)
    return yaw,d

def lms2xy(yaw,d):
    points = np.zeros([len(yaw),2])
    points[:,0] = d*np.cos(yaw)
    points[:,1] = d*np.sin(yaw)
    return points


def mark_all_spawn_points(world):
    _map = world.get_map()
    spawn_points = _map.get_spawn_points()
    for i,w in enumerate(spawn_points):
        world.debug.draw_string(w.location, text = str(i), life_time = 1000)
     

def raw_data2points(lidar_frame):
    if "0.9.8" or "0.9.9.2" in carla.__path__[0]:
        points = np.frombuffer(lidar_frame.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
    elif "0.9.11" in carla.__path__[0]:
        points = np.frombuffer(lidar_frame.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points = points[:,:3]

        #(x,y,z)
        pitch = np.radians(lidar_frame.transform.rotation.pitch)
        pitch_matrix = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
        points = np.matmul(points,pitch_matrix)
        
        #x-axis is along the vehicle
        roll = np.radians(lidar_frame.transform.rotation.roll)
        roll_matrix = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
        points = np.matmul(points,roll_matrix)
        

#   
    return points


def draw_lidar_from_raw(surface, lidar_frame, blend=False, blit = True):
    DIM0 = surface.get_width()
    DIM1 = surface.get_height()
    points = raw_data2points(lidar_frame)
#    rescue_lms(points,12)
    lidar_data = np.array(points[:, :2])
    lidar_data *= DIM1 / 100.0
    lidar_data += (0.5 * DIM0, 0.5 * DIM1)
    lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
    lidar_data = lidar_data.astype(np.int32)
    lidar_data = np.reshape(lidar_data, (-1, 2))
    lidar_img_size = (DIM0, DIM1, 3)
    lidar_img = np.zeros(lidar_img_size)
    lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
    image_surface = pygame.surfarray.make_surface(lidar_img)
    if blend:
        image_surface.set_alpha(100)
        
    if blit == True:
        surface.blit(image_surface, (0, 0))
    return image_surface


def spawn_v_try_with_transforms(client,transforms,number_of_vehicles,autopilot = True):
    vehicles_list = []
    world = client.get_world()
    blueprints = world.get_blueprint_library().filter('vehicle.*')

    #safe?
    blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
    blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
    blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]

    # @todo cannot import these directly.
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    # --------------
    # Spawn vehicles
    # --------------
    batch = []
    for n, transform in enumerate(transforms):
        if n >= number_of_vehicles:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')
#        if autopilot == True:
#            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))
#        else:
#            batch.append(SpawnActor(blueprint, transform))
        

        batch.append(SpawnActor(blueprint, transform))

    for response in client.apply_batch_sync(batch):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)
 
    for v in world.get_actors(vehicles_list):
        physics_control = v.get_physics_control()
        physics_control.final_ratio = 0.2
        v.apply_physics_control(physics_control)
        if autopilot == True:
            v.set_autopilot(True)
#    
    return vehicles_list

def spawn_v_try(client,number_of_vehicles,number_of_bikes = 0,autopilot = True,port=8000):
    vehicles_list = []
    world = client.get_world()
    blueprints = world.get_blueprint_library().filter('vehicle.*')

    #safe?
    blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
    blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
    blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)
    random.seed(0)
    if number_of_vehicles < number_of_spawn_points:
        pass
        random.shuffle(spawn_points)
    elif number_of_vehicles > number_of_spawn_points:
        msg = 'requested %d vehicles, but could only find %d spawn points'
        logging.warning(msg, number_of_vehicles, number_of_spawn_points)
        number_of_vehicles = number_of_spawn_points

    # @todo cannot import these directly.
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    batch = []
    # --------------
    # Spawn vehicles
    # --------------
    for n, transform in enumerate(spawn_points):
        if n >= number_of_vehicles:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')
        
        
#        vehicle = world.try_spawn_actor(blueprint, transform)
#        vehicles_list.append(vehicle)        
        
        
        if autopilot == True:
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, port)))
        else:
            batch.append(SpawnActor(blueprint, transform))


    """
    # -------------
    # Spawn Walkers
    # -------------
    # 1. take all the random locations to spawn
    spawn_points = []
    for i in range(args.number_of_walkers):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
            spawn_points.append(spawn_point)
    # 2. we spawn the walker object
    batch = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprintsWalkers)
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        batch.append(SpawnActor(walker_bp, spawn_point))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id
    # 4. we put altogether the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)

    # wait for a tick to ensure client receives the last transform of the walkers we have just created
    world.wait_for_tick()

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    for i in range(0, len(all_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point
        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        # random max speed
        all_actors[i].set_max_speed(1 + random.random())    # max speed between 1 and 2 (default is 1.4 m/s)

    print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

    while True:
        world.wait_for_tick()
    """
#    if autopilot == True:
#        for v in vehicles_list:
#            v.set_autopilot(True)
    #spawn bikes
    # --------------
    # Spawn cola
    # --------------
#    blueprints = world.get_blueprint_library().filter('vehicle.carlamotors.*')
#    
#    for n, transform in enumerate(spawn_points[number_of_vehicles:]):
#        if n >= number_of_bikes:
#            break
#        blueprint = random.choice(blueprints)
#        if blueprint.has_attribute('color'):
#            color = random.choice(blueprint.get_attribute('color').recommended_values)
#            blueprint.set_attribute('color', color)
#        if blueprint.has_attribute('driver_id'):
#            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
#            blueprint.set_attribute('driver_id', driver_id)
#        blueprint.set_attribute('role_name', 'autopilot')
#        
#        
##        vehicle = world.try_spawn_actor(blueprint, transform)
##        vehicles_list.append(vehicle)        
#        
#        
#        if autopilot == True:
#            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True,port)))
#        else:
#            batch.append(SpawnActor(blueprint, transform))    
    #spawn bikes
    # --------------
    # Spawn vehicles
    # --------------
    if number_of_bikes > 0:
        blueprints = world.get_blueprint_library().filter('vehicle.bh.crossbike')
        
        for n, transform in enumerate(spawn_points[number_of_vehicles+1:]):
            if n >= number_of_bikes:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            
            
    #        vehicle = world.try_spawn_actor(blueprint, transform)
    #        vehicles_list.append(vehicle)        
            
            
            if autopilot == True:
                batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, port)))
            else:
                batch.append(SpawnActor(blueprint, transform))            
    
    for response in client.apply_batch_sync(batch):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)

    for v in world.get_actors(vehicles_list):
        physics_control = v.get_physics_control()
        physics_control.final_ratio = 0.2
        v.apply_physics_control(physics_control)


    return vehicles_list


def spawn_v_finally(client,vehicles_list):

    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
    print('\ndestroying %d vehicles' % len(vehicles_list))
    """
    # stop walker controllers (list is [controler, actor, controller, actor ...])
    for i in range(0, len(all_id), 2):
        all_actors[i].stop()

    print('\ndestroying %d walkers' % len(walkers_list))
    client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
    """
