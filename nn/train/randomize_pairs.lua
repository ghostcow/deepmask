RandomizePairs = {}

function RandomizePairs.randomizeImages(labels)

    -- prepare a map between each label to it's images
    nImages = labels:size(1)
    shuffleImages = torch.randperm(nImages)
    mapLabelToImages = {}
    for iImage = 1,nImages do
        label = labels[shuffleImages[iImage]]
        if not mapLabelToImages[label] then
            mapLabelToImages[label] = {}
        end
        table.insert(mapLabelToImages[label], shuffleImages[iImage])
    end

    -- randomize pairs
    shuffle = torch.Tensor(2*torch.floor(nImages/2)):fill(0)
    shufflePersons = torch.randperm(#mapLabelToImages)
    isPositive = true -- should take positive pair as the next one
    reshuffle = false
    iLabel = 1
    iImage = 1

    while true do
        -- stop criteria : no more images
        if (#mapLabelToImages == 0) then
            break
        end

        if (#mapLabelToImages == 1) then
            -- we are left with only one label
            while (#mapLabelToImages[1] >= 2) do
                pos_i1 = table.remove(mapLabelToImages[1])
                pos_i2 = table.remove(mapLabelToImages[1])
                -- print('+1', labels[pos_i1], labels[pos_i2], ':', pos_i1, pos_i2)
                shuffle[iImage] = pos_i1
                shuffle[iImage+1] = pos_i2
                iImage = iImage + 2
            end
            break
        end

        label1 = shufflePersons[iLabel]
        if isPositive and (#mapLabelToImages[label1] >= 2) then
            -- positive pair
            pos_i1 = table.remove(mapLabelToImages[label1])
            pos_i2 = table.remove(mapLabelToImages[label1])
            shuffle[iImage] = pos_i1
            shuffle[iImage+1] = pos_i2
            iImage = iImage + 2
            if (#mapLabelToImages[label1] == 0) then
                table.remove(mapLabelToImages, label1)
                reshuffle = true
            end
            iLabel = iLabel + 1
            if (iLabel > #mapLabelToImages) then
                iLabel = 1
            end
            isPositive = false
        else
            -- negative pair
            neg_i1 = table.remove(mapLabelToImages[label1])
            if (#mapLabelToImages[label1] == 0) then
                table.remove(mapLabelToImages, label1)
                reshuffle = true
                -- we don't need to increment iLabel,
                -- because we removed the appropriate cell from mapLabelToImages
            else
                iLabel = iLabel + 1
            end
            if (iLabel > shufflePersons:size(1)) then
                iLabel = 1
            end

            label2 = shufflePersons[iLabel]
            while (label2 > #mapLabelToImages) do
                -- because we possibly removed label from mapLabelToImages,
                -- shufflePersons might be unupdated
                iLabel = iLabel + 1
                if (iLabel > shufflePersons:size(1)) then
                    iLabel = 1
                end
                label2 = shufflePersons[iLabel]
            end

            neg_i2 = table.remove(mapLabelToImages[label2])
            if (#mapLabelToImages[label2] == 0) then
                table.remove(mapLabelToImages, label2)
                reshuffle = true
            end
            shuffle[iImage] = neg_i1
            shuffle[iImage+1] = neg_i2
            iImage = iImage + 2
            isPositive = true
        end

        if reshuffle then
            if (#mapLabelToImages > 0) then
                shufflePersons = torch.randperm(#mapLabelToImages)
            end
            iLabel = 1
        end
    end

    return shuffle
end