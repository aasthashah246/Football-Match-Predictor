
    else:
        prediction = "Draw"
    #confidence = np.max(model.predict_proba(input_data)) * 100
    probabilities = model.predict_proba(input_data)[0]

    print(probabilities)    

    confidence = np.max(probabilities) * 100
    print("Classes:", model.classes_)
    print("Probabilities:", probabilities)
    print("Prediction:", prediction)
