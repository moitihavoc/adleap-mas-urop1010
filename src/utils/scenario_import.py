# Import/Generate scenarios components for Level-based Foraging scenarios 
# via images reading
# - White pixels == free spaces
# - Black pixels == obstacles
# - Red pixels == agents
# - Blue pixels == tasks

from PIL import Image

# settings
your_image_path = './imgs/imported_scenarios/theoffice.png'
visibility = 'partial'

image = Image.open(your_image_path)
image_pixels = image.load()

components = {}
components['dim'] = (int(image.size[0]),int(image.size[1]))
components['visibility'] = visibility
components['agents'] = []
components['adhoc_agent_index'] = '0'
components['tasks'] = []
components['obstacles'] = []

agent_idx, task_idx = 0, 0
for x in range(image.size[0]):
    for y in range(image.size[1]):
        pix = image_pixels[x,y] if len(image_pixels[x,y]) == 3 \
            else (image_pixels[x,y][0],image_pixels[x,y][1],image_pixels[x,y][2])
        if pix == (255,0,0):
            components['agents'].append( \
                'Agent(index=\''+str(agent_idx)+'\','+\
                'atype=method,'+\
                'position=('+str(x)+','+str(y)+'),'+\
                'direction=1*np.pi/2,radius=0.2,angle=0.3,level=1.0),' 
            )
            agent_idx += 1
        elif pix == (0,0,255):
            components['tasks'].append( \
                'Task(index=\''+str(task_idx)+'\','+\
                'position=('+str(x)+','+str(y)+'),level=1.0),')
            task_idx += 1
        elif pix == (0,0,0): 
            components['obstacles'].append((x,y))

print('{')
print('\t\'dim\': '+str(components['dim'])+',')
print('\t\'visibility\': \''+str(components['visibility'])+'\',')
print('\t\'agents\': [')
for a in components['agents']:
    print('\t\t',a)
print('\t],')
print('\t\'adhoc_agent_index\': \''+str(components['adhoc_agent_index'])+'\',')
print('\t\'tasks\': [')
for t in components['tasks']:
    print('\t\t',t)
print('\t],')
print('\t\'obstacles\': [')
for o in components['obstacles']:
    print(o,',',end=' ')
print('\n\t],')
print('}')