import matplotlib.pyplot as plt

def plot_times_values(times, values,labels):
    plt.plot(times, values, 'o', label=labels)
    for i, (time, value, label) in enumerate(zip(times, values, labels)):
        plt.text(time, value, label, ha='center', va='bottom')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig("TimeGraph.png")
# Example usage
times = [167.2,213.8,207.8,1209,467.2,797.5,1180,1694.8]
values = [-1.18,-1.18,-1.85,-1.85,-1.85,-1.85,-1.85,-1.85]
labels = [2,5,10,20,30,50,100,150]
times = [t/l for t,l in zip(times,labels)]
plot_times_values(times, values,labels)