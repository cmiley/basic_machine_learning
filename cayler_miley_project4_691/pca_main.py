import compress as cm

if __name__ == '__main__':
    X = cm.load_data('Data/Train/')
    cm.compress_images(X, 400)
