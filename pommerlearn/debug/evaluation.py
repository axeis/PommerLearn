import numpy as np
import zarr
import matplotlib.pyplot as plt


from dataset_util import get_agent_died_in_step

def deaths_per_step(z):

    deaths = []

    print(deaths)
    for episode in range(0, len(z.attrs.get('EpisodeSteps'))):
        
        dead_in_steps = get_agent_died_in_step(z.attrs.get('EpisodeActions')[episode],
                                z.attrs.get('EpisodeDead')[episode]
        )
        deaths = np.append(deaths, dead_in_steps)
        # print("Episode: {}".format(episode))
        # print(dead_in_steps)


    print(deaths)

    plt.plot(deaths)
    plt.show()

    return



def main():
    z = zarr.open('/PommerLearn/1M_simple_0.zr', 'r')

    print("Info:")
    print(z.info)

    deaths_per_step(z)





if __name__ == '__main__':
    main()
