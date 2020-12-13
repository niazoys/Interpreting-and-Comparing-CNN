



#####################################################  Niaz ######################################################
# def load_data(batch_size):
  
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    trainset =datasets.CIFAR10(
    root='./PATH', train=True, download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    testset = datasets.CIFAR10(
    root='./PATH', train=False, download=True,transform=transform)
    testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=1)

    return trainloader,testloader,trainset

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loader,_,trainset =load_data(1000)
    epochs= 1
    # for epoch in range(epochs):
    #     for batch_idx, (data, target) in enumerate(loader):
    #         optimizer.zero_grad()
    #         output = model(data)
            # loss = criterion(output, data)
            # loss.backward()
            # optimizer.step()
            
            # print('Epoch {}, Batch idx {}, loss {}'.format(
            #     epoch, batch_idx, loss.item()))







    # # # Plot some images
    # idx = torch.randint(0, output.size(0), ())
    # pred = normalize_output(output[idx, 0])
    # img = data[idx, 0]

    # fig, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(img.detach().numpy())
    # axarr[1].imshow(pred.detach().numpy())



    #plot the feature maps
    #vis.plot_single_featuremaps(interm_output[output_layers[0]],unit_num=25,name=output_layers[0],color_map='color',savefig=True)
    
    #save the activation for all the images 
    #torch.save(activation_image_list, 'D:\\Net\\NetDissect\\dataset\\broden1_227\\Activation_resnet18.pt')
    



    
    ######################## Captum Network IG ###############

    # ig = IntegratedGradients(model)
    # ig_attribution=ig.attribute(x,target=preds)
    # ig_attribution=np.transpose(ig_attribution.detach().numpy().squeeze(),(1,2,0))


    # _ = vizu.visualize_image_attr(ig_attribution, (original_image*0)+255, method="blended_heat_map",sign="all",
    #                 show_colorbar=True, title="Overlayed Integrated Gradients")

    ############################# Captum Neuron IG  #####################
    # neuron_ig = NeuronIntegratedGradients(self.model,layer)
    # attribution = neuron_ig.attribute(x, (4,5,2))  

    ################## with out captum#####################
    # Vanilla backprop
    # IG = IntegratedGradients(self.model) 

    # # Generate gradients
    # integrated_grads = IG.generate_integrated_gradients(x, preds, 100)
    
    # # Convert to grayscale
    # grayscale_integrated_grads = convert_to_grayscale(integrated_grads)
    
    # plt.imshow(grayscale_integrated_grads)
    # plt.show()
    # Save grayscale gradients
    # save_gradient_images(grayscale_integrated_grads, file_name_to_export + '_Integrated_G_gray')
    # print('Integrated gradients completed.')

#################################################################################################################




############################################Zahid################################################################
