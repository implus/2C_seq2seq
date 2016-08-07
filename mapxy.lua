function randommapxy(vocab, base)
    mapx = {} mapy = {}
    local ind = {}
    for i = 1, base do
        for j = 1, base do
            table.insert(ind, {i, j})
        end
    end
    local r = torch.randperm(base * base)
    for w = 1, vocab do
        local p = ind[r[w]]
        mapx[w] = p[1]
        mapy[w] = p[2]
    end
    return mapx, mapy
end

function check_conflict(mapx0, mapy0, vocab, base)
    print('check if conflict with mapx, mapy')
    allset = {}
    distinct = {}
    for i = 1, vocab do
        if mapx0[i] <= 0 or mapx0[i] > base then print('error! range out 0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        elseif mapy0[i] <= 0 or mapy0[i] > base then print ('error ! range out 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') end

        newval = base * (mapx0[i] - 1) + mapy0[i] - 1
        if allset[newval] == nil then table.insert(distinct, newval) end
        allset[newval] = 1
    end
    if #distinct ~= vocab then
        print('error!!!!!!!!!!!!!!!!!!!! #alldistinct = ', #distinct, 'not match with vocab = ', vocab)
    end
end

function mapxyfromfile(file, base)
    local filemapxy = io.open(file, 'r')
    local idx = 1 local idy
    mapx = {} mapy = {}
    while true do
        local str = filemapxy:read()
        if str == nil then break end
        str = stringx.split(str)
        for idy = 1, base do
            local word = tonumber(str[idy])
            if word > 0 then
                mapx[word] = idx
                mapy[word] = idy
            end
        end
        idx = idx + 1
    end
    return mapx, mapy
end
