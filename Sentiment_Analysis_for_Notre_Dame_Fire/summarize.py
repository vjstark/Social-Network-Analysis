import pickle

def main():
    f = open('summary.txt','w')
    with open('data_from_clusterpy','rb') as fc:
        content = pickle.load(fc)
        for i in content:
            print(i, file = f)
    with open('data_from_classifypy','rb') as fc:
        content = pickle.load(fc)
        for i in content:
            print(i, file = f)
    f.close()

if __name__ == '__main__':
    main()
