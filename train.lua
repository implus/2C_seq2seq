require 'nn'
require 'nngraph'
require 'hdf5'

require 'data.lua'
require 'util.lua'
require 'mapxy.lua'
require 'models.lua'
require 'model_utils.lua'

cmd = torch.CmdLine()
-- data files

cmd:text('') cmd:text("**Data options") cmd:text('')
cmd:option('-data_file', 'data/demo-train.hdf5', [[train]])
cmd:option('-val_data_file', 'data/demo-val.hdf5', [[valid]])
cmd:option('-savefile', 'models/2C_seq2seq', [[savefile_epochX_PPL.t7]])
cmd:option('-train_from', '', [[from some savefile]])

-- rnn model
cmd:text('') cmd:text('**Mode options') cmd:text('')
cmd:option('-num_layers', 2, [[Number of layers in enc/dec]])
cmd:option('-rnn_size', 650, [[rnn size]])
cmd:option('-word_vec_size', 650, [[word embedding size]])
cmd:option('-attn', 1, [[use attention on the decoder side]])
cmd:option('-brnn', 1, [[use bidirectional RNN]])

-- optimization
cmd:text('') cmd:text('**Optimization options') cmd:text('')
cmd:option('-epochs', 13, [[epochs]])
cmd:option('-start_epoch', 1, [[loading from a checkpoint, start where]])
cmd:option('-param_init', 0.1, [[uniform distribution]])
cmd:option('-optim', 'sgd', [[sgd, adagrad, adadelta, adam]])
cmd:option('-learning_rate', 1, [[learning_rate]])
cmd:option('-max_grad_norm', 5, [[norm of gradients]])
cmd:option('-dropout', 0.3, [[dropout rate]])
cmd:option('-lr_decay', 0.5, [[decay lr]])
cmd:option('-start_decay_at', 9, [[start]])
cmd:option('-curriculum', 0, [[order minibatches]])
cmd:option('-max_batch_l', '', [[if blank infer valid max batch size]])

-- we do n't load word vec or fix word vec

-- GPU
cmd:option('-gpuid', 1, [[enc, -1 use CPU]])
cmd:option('-gpuid2', 2, [[dec, -1 use CPU]])
cmd:option('-cudnn', 1, [[use cudnn]])

-- bookkeeping
cmd:option('-save_every', 1, [[save epoch]])
cmd:option('-save_every_step', 10000, [[save epoch step]])
cmd:option('-print_every', 100, [[minibatches print]])
cmd:option('-seed', 19941229, [[random seed]])
cmd:option('-debug', 0, [[if debug]])

-- src_mapxy, trg_mapxy
cmd:option('-src_mapxy', '', [[src_mapxy file]])
cmd:option('-trg_mapxy', '', [[trg_mapxy file]])

opt = cmd:parse(arg)
flag = {}
if opt.gpuid > 0 then flag.oneGPU = true else flag.oneGPU = false end
if opt.gpuid2 > 0 then flag.twoGPU = true else flag.twoGPU = false end
if opt.gpuid2 == nil or opt.gpuid2 <= 0 then opt.gpuid2 = opt.gpuid print('force the same') end -- force to the same gpu
function setGPU(id) 
    if id > 0 then cutorch.setDevice(id) end
end
torch.manualSeed(opt.seed)


-- zero table according to its GPU
function zero_table(t, t_gpu)
    for i = 1, #t do
        setGPU(t_gpu[i])
        t[i]:zero()
    end
end

function train(train_data, valid_data)
    local timer = torch.Timer()
    local num_params = 0
    local start_decay = 0
    params, grad_params = {}, {}
    opt.train_perf = {}
    opt.val_perf   = {}

    -- init all params, get all params and grad
    for i = 1, #layers do -- {encoder, decoder, generatorx, generatory, enc_embedx, enc_embedy, dec_embedx, dec_embedy, encoder_bwd}
                          -- please add layers_GPU for setting
        setGPU(layers_GPU[i])
        local p, gp = layers[i]:getParameters()
        if opt.train_from:len() == 0 then 
            p:uniform(-opt.param_init, opt.param_init)
        end
        num_params = num_params + p:size(1)
        params[i] = p
        grad_params[i] = gp
    end
    
    print('number of parameters: '..num_params)
    
    -- assign volumns for attention gradients bp from decoder to encoder
    encoder_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l * 2, opt.rnn_size)
    encoder_bwd_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l * 2, opt.rnn_size)
    context_proto      = torch.zeros(opt.max_batch_l, opt.max_sent_l * 2, opt.rnn_size)
    -- need more copies of the above if using two gpus
    if flag.twoGPU then
        encoder_grad_proto2 = torch.zeros(opt.max_batch_l, opt.max_sent_l * 2, opt.rnn_size)
        encoder_bwd_grad_proto2 = torch.zeros(opt.max_batch_l, opt.max_sent_l * 2, opt.rnn_size)
        context_proto2      = torch.zeros(opt.max_batch_l, opt.max_sent_l * 2, opt.rnn_size)
    end

    -- clone encoder/decoder up to max source/target length x 2 for 2C_seq2seq
    encoder_clones = clone_many_times(encoder, opt.max_sent_l_src *  2)
    decoder_clones = clone_many_times(decoder, opt.max_sent_l_targ * 2) -- !!!!!! pay attention x 2 for 2C_seq2seq
    if opt.brnn == 1 then
        encoder_bwd_clones = clone_many_times(encoder_bwd, opt.max_sent_l_src * 2)
    end
    
    -- ??? do n't know why we need to setReuse()
    function msr(m) m:setReuse() end
    for i = 1, opt.max_sent_l_src * 2 do
        if encoder_clones[i].apply then
            encoder_clones[i]:apply(msr)
        end
        if opt.brnn == 1 then
            encoder_bwd_clones[i]:apply(msr)
        end
    end
    for i = 1, opt.max_sent_l_targ * 2 do
        if decoder_clones[i].apply then
            decoder_clones[i]:apply(msr)
        end
    end

    
    print('h_init go begin')
    print(opt.max_batch_l, opt.rnn_size)
    local h_init = torch.zeros(opt.max_batch_l, opt.rnn_size)
    setGPU(opt.gpuid)
    h_init = h_init:cuda()
    if flag.twoGPU then
        encoder_grad_proto2     = encoder_grad_proto2:cuda()
        encoder_bwd_grad_proto2 = encoder_bwd_grad_proto2:cuda()
        context_proto           = context_proto:cuda()

        setGPU(opt.gpuid2)
        encoder_grad_proto      = encoder_grad_proto:cuda()
        encoder_bwd_grad_proto  = encoder_bwd_grad_proto:cuda()
        context_proto2          = context_proto2:cuda()
    else
        context_proto           = context_proto:cuda()
        encoder_grad_proto      = encoder_grad_proto:cuda()
        encoder_bwd_grad_proto  = encoder_bwd_grad_proto:cuda()
    end

    -- these are initial states of encoder/decoder for fwd/bwd steps
    setGPU(opt.gpuid)
    print('set gpu1, these are initial states of encoder/decoder for fwd/bwd steps')
    init_fwd_enc = {}
    init_bwd_enc = {}
    init_fwd_dec = {}
    init_bwd_dec = {}
    init_tmp1    = {}
    init_tmp2    = {}

    for L = 1, opt.num_layers do
        table.insert(init_fwd_enc, h_init:clone())
        table.insert(init_fwd_enc, h_init:clone()) -- cell and hidden
        table.insert(init_bwd_enc, h_init:clone())
        table.insert(init_bwd_enc, h_init:clone())
        table.insert(init_tmp1,    h_init:clone())
        table.insert(init_tmp1,    h_init:clone())
    end

    setGPU(opt.gpuid2)
    for L = 1, opt.num_layers do
        table.insert(init_fwd_dec, h_init:clone())
        table.insert(init_fwd_dec, h_init:clone())
        table.insert(init_bwd_dec, h_init:clone())
        table.insert(init_bwd_dec, h_init:clone())
        table.insert(init_tmp2,    h_init:clone())
        table.insert(init_tmp2,    h_init:clone())
    end
    table.insert(init_bwd_dec, h_init:clone())     -- for label backward !
    
    dec_offset = 3 -- because inputs for dec, cell hidden starts from dec_offset

    function reset_state(state, batch_l, t)
        if opt.debug then print('reset state begin') print(state, batch_l, t) end
        if t == nil then
            local u = {}
            for i = 1, #state do
                state[i]:zero() table.insert(u, state[i][{{1, batch_l}}])
            end
            return u
        else
            local u = {[t] = {}}
            for i = 1, #state do
                state[i]:zero() table.insert(u[t], state[i][{{1, batch_l}}])
            end
            return u
        end
        if opt.debug then print('reset state end') print() end
    end

    -- clean layer before saving to make the model smaller
    function clean_layer(layer)
        if layer == nil then return end
        if flag.oneGPU then
            layer.output = torch.CudaTensor() layer.gradInput = torch.CudaTensor()
        else
            layer.output = torch.DoubleTensor() layer.gradInput = torch.DoubleTensor()
        end
        if layer.modules then
            for i, mod in ipairs(layer.modules) do
                clean_layer(mod)
            end
        elseif torch.type(self) == 'nn.gModule' then
            layer:apply(clean_layer)
        end
    end

    -- decay lr
    function decay_lr(epoch)
        print(opt.val_perf)
        if epoch >= opt.start_decay_at then start_decay = 1 end
        if opt.val_perf[#opt.val_perf] ~= nil and opt.val_perf[#opt.val_perf - 1] ~= nil then
            local curr_ppl = opt.val_perf[#opt.val_perf]
            local prev_ppl = opt.val_perf[#opt.val_perf - 1]
            if curr_ppl > prev_ppl then start_decay = 1 end
        end
        if start_decay == 1 then opt.learning_rate = opt.learning_rate * opt.lr_decay end
    end

    function map(x, m, gpu)
        if opt.debug then print('map begin') end
        setGPU(gpu)
        local y = x:clone()
        local s = y:storage()
        for i = 1, s:size() do
            s[i] = m[s[i]]
        end
        if opt.debug then print('map end') end
        return y
    end

    function train_batch(data, epoch)
        collectgarbage()
        local train_nonzeros    = 0
        local train_loss        = 0
        local batch_order       = torch.randperm(data.length)
        local start_time        = timer:time().real
        local num_words_target  = 0
        local num_words_source  = 0

        local toend
        if opt.debug then toend = 1 else toend = data:size() end
        for i = 1, toend do
            zero_table(grad_params, layers_GPU)
            local d
            if epoch <= opt.curriculum then d = data[i]
            else d = data[batch_order[i]] end
            local batch_l, target_l, source_l = d[5], d[6], d[7]
            local target, target_out, nonzeros_filter, source = d[1], d[2], d[3], d[4]
            local tmp = nonzeros_filter:clone():double()
            if opt.debug then print(tmp) end
            target = target_out local nonzeros = torch.sum(tmp) -- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! very important !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            local encoder_grads = encoder_grad_proto[{{1, batch_l}, {1, source_l * 2}}]
            local encoder_bwd_grads
            if opt.brnn == 1 then encoder_bwd_grads = encoder_bwd_grad_proto[{{1, batch_l}, {1, source_l * 2}}] end
            if flag.oneGPU then setGPU(opt.gpuid) end

            source_x = map(source, src_mapx, opt.gpuid)
            source_y = map(source, src_mapy, opt.gpuid)
            target_x = map(target, trg_mapx, opt.gpuid2)
            target_y = map(target, trg_mapy, opt.gpuid2)
            setGPU(opt.gpuid)

            local rnn_state_enc = reset_state(init_fwd_enc, batch_l, 0)
        
            local context = context_proto[{{1, batch_l}, {1, source_l * 2}}]
    -- {encoder, decoder, generatorx, generatory, enc_embedx, enc_embedy, dec_embedx, dec_embedy, encoder_bwd}
            for t = 1, source_l do
                enc_embedx:training()
                encoder_clones[2 * t - 1]:training() -- embed x
                -- first forward embedding x
                local source_embedx = enc_embedx:forward(source_x[t])
                local encoder_input_x = {source_embedx, table.unpack(rnn_state_enc[2 * t - 2])}
                local out = encoder_clones[2 * t - 1]:forward(encoder_input_x)
                rnn_state_enc[2 * t - 1] = out
                context[{{}, 2 * t - 1}]:copy(out[#out])

                enc_embedy:training()
                encoder_clones[2 * t]:training()     -- embed y
                -- first forward embedding y
                local source_embedy = enc_embedy:forward(source_y[t])
                local encoder_input_y = {source_embedy, table.unpack(rnn_state_enc[2 * t - 1])}
                local out = encoder_clones[2 * t]:forward(encoder_input_y)
                rnn_state_enc[2 * t] = out
                context[{{}, 2 * t}]:copy(out[#out])
            end

            local rnn_state_enc_bwd
            if opt.brnn == 1 then
                rnn_state_enc_bwd = reset_state(init_fwd_enc, batch_l, 2 * source_l + 1)
                for t = source_l, 1, -1 do
                    enc_embedy:training()
                    encoder_bwd_clones[2 * t]:training()
                    -- first forward embedding y
                    local source_embedy = enc_embedy:forward(source_y[t])
                    local encoder_input_y = {source_embedy, table.unpack(rnn_state_enc_bwd[2 * t + 1])}
                    local out = encoder_bwd_clones[2 * t]:forward(encoder_input_y)
                    rnn_state_enc_bwd[2 * t] = out
                    context[{{}, 2 * t}]:add(out[#out])

                    enc_embedx:training()
                    encoder_bwd_clones[2 * t - 1]:training()
                    -- second forward embedding x
                    local source_embedx = enc_embedx:forward(source_x[t])
                    local encoder_input_x = {source_embedx, table.unpack(rnn_state_enc_bwd[2 * t])}
                    local out = encoder_bwd_clones[2 * t - 1]:forward(encoder_input_x)
                    rnn_state_enc_bwd[2 * t - 1] = out
                    context[{{}, 2 * t - 1}]:add(out[#out])
                end
            end

            -- go to next gpuid2
            if flag.twoGPU then
                setGPU(opt.gpuid2)
                local context2 = context_proto2[{{1, batch_l}, {1, source_l * 2}}] -- in gpu2
                context2:copy(context) -- from gpu1 to gpu2
                context = context2     -- reference
            end

            -- copy encoder last hidden state as decoder initial state
            if opt.debug then print('copy encoder last hidden state as decoder initial state') end
            local rnn_state_dec = reset_state(init_fwd_dec, batch_l, 0)
            for L = 1, opt.num_layers do
                rnn_state_dec[0][L * 2 - 1]:copy(rnn_state_enc[source_l * 2][L * 2 - 1])
                rnn_state_dec[0][L * 2    ]:copy(rnn_state_enc[source_l * 2][L * 2    ])
            end
            if opt.brnn == 1 then
                local tmp_state = reset_state(init_tmp2, batch_l)
                for L = 1, opt.num_layers do
                    tmp_state[L * 2 - 1]:copy(rnn_state_enc_bwd[1][L * 2 - 1])
                    tmp_state[L * 2    ]:copy(rnn_state_enc_bwd[1][L * 2    ])
                end
                for L = 1, opt.num_layers do
                    --rnn_state_dec[0][L * 2 - 1]:add(rnn_state_enc_bwd[1][L * 2 - 1])
                    --rnn_state_dec[0][L * 2    ]:add(rnn_state_enc_bwd[1][L * 2    ])
                    rnn_state_dec[0][L * 2 - 1]:add(tmp_state[L * 2 - 1])
                    rnn_state_dec[0][L * 2    ]:add(tmp_state[L * 2    ])
                end
            end

            -- define first input all 0 embedding
            if opt.debug then print('define first input all 0 embedding') end
            local decoder_input_zero = torch.zeros(batch_l, opt.word_vec_size):cuda()
            -- forward prop decoder
            local preds = {}
            for t = 1, target_l do
                dec_embedy:training()
                decoder_clones[2 * t - 1]:training()
                -- forward y to predict x
                local decoder_input_y, first, ctx
                if t == 1 then          first = decoder_input_zero
                else                    first = dec_embedy:forward(target_y[t - 1]) end
                if opt.attn == 1 then   ctx   = context
                else                    ctx   = context[{{}, source_l * 2}] end
                decoder_input_y = {first, ctx, table.unpack(rnn_state_dec[2 * t - 2])}
                local out = decoder_clones[2 * t - 1]:forward(decoder_input_y)
                local next_state = {}
                table.insert(preds, out[#out]) -- pred x
                for j = 1, #out - 1 do table.insert(next_state, out[j]) end
                rnn_state_dec[2 * t - 1] = next_state

                dec_embedx:training()
                decoder_clones[2 * t]:training()
                -- forward x to predict y
                local decoder_input_x = {dec_embedx:forward(target_x[t]), ctx, table.unpack(rnn_state_dec[2 * t - 1])}
                local out = decoder_clones[2 * t]:forward(decoder_input_x)
                local next_state = {}
                table.insert(preds, out[#out]) -- pred y
                for j = 1, #out - 1 do table.insert(next_state, out[j]) end
                rnn_state_dec[2 * t] = next_state
            end

            -- backward prop decoder
            if opt.debug then print('backward prop decoder') end
            encoder_grads:zero()
            if opt.brnn == 1 then encoder_bwd_grads:zero() end

            local drnn_state_dec = reset_state(init_bwd_dec, batch_l)
            local loss = 0
            for t = target_l, 1, -1 do
                local ctx
                if opt.attn == 1 then   ctx   = context
                else                    ctx   = context[{{}, source_l * 2}] end

                local predy = generatory:forward(preds[2 * t]) -- backward y
                loss = loss + criterion:forward(predy, target_y[t])/batch_l -- no need for target_out
                local dl_dpred = criterion:backward(predy, target_y[t])
                if opt.debug then print(dl_dpred:size(), 'vs', nonzeros_filter[t]:size()) end
                dl_dpred:div(batch_l)
                --dl_dpred:cmul(nonzeros_filter[t]:view(dl_dpred:size(1), 1):expand(dl_dpred:size(1), dl_dpred:size(2)))
                local dl_dtarget = generatory:backward(preds[2 * t], dl_dpred)
                drnn_state_dec[#drnn_state_dec]:add(dl_dtarget)

                local decoder_input_x = {dec_embedx:forward(target_x[t]), ctx, table.unpack(rnn_state_dec[2 * t - 1])}
                local dlst = decoder_clones[2 * t]:backward(decoder_input_x, drnn_state_dec)
                dec_embedx:backward(target_x[t], dlst[1])
                if opt.attn == 1 then
                    encoder_grads:add(dlst[2])
                    if opt.brnn == 1 then encoder_bwd_grads:add(dlst[2]) end
                else
                    encoder_grads[{{}, source_l * 2}]:add(dlst[2])
                    if opt.brnn == 1 then encoder_bwd_grads[{{}, 1}]:add(dlst[2]) end
                end
                drnn_state_dec[#drnn_state_dec]:zero()
                for j = dec_offset, #dlst do drnn_state_dec[j - dec_offset + 1]:copy(dlst[j]) end

                -- fill drnn_state_dec[#drnn_state_dec] with generator error
                local predx = generatorx:forward(preds[2 * t - 1]) -- backward x
                loss = loss + criterion:forward(predx, target_x[t])/batch_l
                local dl_dpred = criterion:backward(predx, target_x[t])
                dl_dpred:div(batch_l)
                --dl_dpred:cmul(nonzeros_filter[t]:view(dl_dpred:size(1), 1):expand(dl_dpred:size(1), dl_dpred:size(2)))
                local dl_dtarget = generatorx:backward(preds[2 * t - 1], dl_dpred)
                drnn_state_dec[#drnn_state_dec]:add(dl_dtarget)

                local first
                if t == 1 then          first = decoder_input_zero
                else                    first = dec_embedy:forward(target_y[t - 1]) end
                local decoder_input_y = {first, ctx, table.unpack(rnn_state_dec[2 * t - 2])}
                local dlst = decoder_clones[2 * t - 1]:backward(decoder_input_y, drnn_state_dec)
                if t > 1 then dec_embedy:backward(target_y[t - 1], dlst[1]) end
                if opt.attn == 1 then
                    encoder_grads:add(dlst[2])
                    if opt.brnn == 1 then encoder_bwd_grads:add(dlst[2]) end
                else
                    encoder_grads[{{}, source_l * 2}]:add(dlst[2])
                    if opt.brnn == 1 then encoder_bwd_grads[{{}, 1}]:add(dlst[2]) end
                end
                drnn_state_dec[#drnn_state_dec]:zero()
                for j = dec_offset, #dlst do drnn_state_dec[j - dec_offset + 1]:copy(dlst[j]) end
            end
            
            -- backward prop encoder
            if opt.debug then print('backward prop encoder') end
            if flag.twoGPU then
                setGPU(opt.gpuid)
                local encoder_grads2 = encoder_grad_proto2[{{1, batch_l}, {1, source_l * 2}}]
                encoder_grads2:zero()
                encoder_grads2:copy(encoder_grads)
                encoder_grads = encoder_grads2 -- batch_l x source_l*2 x rnn_size
                local encoder_bwd_grads2 = encoder_bwd_grad_proto2[{{1, batch_l}, {1, source_l * 2}}]
                encoder_bwd_grads2:zero()
                encoder_bwd_grads2:copy(encoder_bwd_grads)
                encoder_bwd_grads = encoder_bwd_grads2
            end

            local drnn_state_enc = reset_state(init_bwd_enc, batch_l)
            for L = 1, opt.num_layers do
                drnn_state_enc[L * 2 - 1]:copy(drnn_state_dec[L * 2 - 1])
                drnn_state_enc[L * 2    ]:copy(drnn_state_dec[L * 2    ])
            end
            for t = source_l, 1, -1 do
                -- backward y
                local encoder_input_y = {enc_embedy:forward(source_y[t]), table.unpack(rnn_state_enc[2 * t - 1])}
                drnn_state_enc[#drnn_state_enc]:add(encoder_grads[{{}, 2 * t}]) -- pay attention we add more calculation for code easy
                local dlst = encoder_clones[2 * t]:backward(encoder_input_y, drnn_state_enc)
                enc_embedy:backward(source_y[t], dlst[1])
                for j = 1, #drnn_state_enc do
                    drnn_state_enc[j]:copy(dlst[j + 1])
                end

                local encoder_input_x = {enc_embedx:forward(source_x[t]), table.unpack(rnn_state_enc[2 * t - 2])}
                drnn_state_enc[#drnn_state_enc]:add(encoder_grads[{{}, 2 * t - 1}]) -- pay attention we add more calculation for code easy
                local dlst = encoder_clones[2 * t - 1]:backward(encoder_input_x, drnn_state_enc)
                enc_embedx:backward(source_x[t], dlst[1])
                for j = 1, #drnn_state_enc do
                    drnn_state_enc[j]:copy(dlst[j + 1])
                end
            end

            if opt.brnn == 1 then
                local drnn_state_enc = reset_state(init_bwd_enc, batch_l)
                for L = 1, opt.num_layers do
                    drnn_state_enc[L * 2 - 1]:copy(drnn_state_dec[L * 2 - 1])
                    drnn_state_enc[L * 2    ]:copy(drnn_state_dec[L * 2    ])
                end
                for t = 1, source_l do
                    local encoder_input_x = {enc_embedx:forward(source_x[t]), table.unpack(rnn_state_enc_bwd[2 * t])}
                    drnn_state_enc[#drnn_state_enc]:add(encoder_bwd_grads[{{}, 2 * t - 1}])
                    local dlst = encoder_bwd_clones[2 * t - 1]:backward(encoder_input_x, drnn_state_enc)
                    enc_embedx:backward(source_x[t], dlst[1])
                    for j = 1, #drnn_state_enc do
                        drnn_state_enc[j]:copy(dlst[j + 1])
                    end

                    local encoder_input_y = {enc_embedy:forward(source_y[t]), table.unpack(rnn_state_enc_bwd[2 * t + 1])}
                    drnn_state_enc[#drnn_state_enc]:add(encoder_bwd_grads[{{}, 2 * t}])
                    local dlst = encoder_bwd_clones[2 * t]:backward(encoder_input_y, drnn_state_enc)
                    enc_embedy:backward(source_y[t], dlst[1])
                    for j = 1, #drnn_state_enc do
                        drnn_state_enc[j]:copy(dlst[j + 1])
                    end
                end
            end

            grad_norm = 0
            for j = 1, #grad_params do
                setGPU(layers_GPU[j])
                grad_norm = grad_norm + grad_params[j]:norm()^2
            end
            grad_norm = grad_norm^0.5

            -- shrink norm and update params
            local param_norm = 0
            local shrinkage = opt.max_grad_norm / grad_norm
            for j = 1, #grad_params do
                setGPU(layers_GPU[j])
                if shrinkage < 1 then grad_params[j]:mul(shrinkage) end

                if opt.optim == 'adagrad' then
                    adagrad_step(params[j], grad_params[j], layer_etas[j], optStates[j])
                elseif opt.optim == 'adadelta' then
                    adadelta_step(params[j], grad_params[j], layer_etas[j], optStates[j])
                elseif opt.optim == 'adam' then
                    adam_step(params[j], grad_params[j], layer_etas[j], optStates[j])
                else
                    params[j]:add(grad_params[j]:mul(-opt.learning_rate))
                end
                param_norm = param_norm + params[j]:norm()^2
            end
            param_norm = param_norm^0.5

            -- bookkeeping
            num_words_target = num_words_target + batch_l * target_l
            num_words_source = num_words_source + batch_l * source_l
            train_nonzeros   = train_nonzeros + nonzeros
            train_loss       = train_loss + batch_l * loss
            local time_taken = timer:time().real - start_time
            if i % opt.print_every == 0 then
                local stats = string.format('Epoch: %d, Batch: %d/%d, Batch size: %d, LR: %.4f, ', 
                epoch, i, data:size(), batch_l, opt.learning_rate)
                stats = stats .. string.format('PPL: %.2f, |Param|: %.2f, |GParam|: %.2f, ',
                math.exp(train_loss/train_nonzeros), param_norm, grad_norm)
                stats = stats .. string.format('Training: %d/%d/%d total/source/target tokens/sec',
                (num_words_target + num_words_source) / time_taken,
                num_words_source / time_taken,
                num_words_target / time_taken)
                print(stats)
            end
            if opt.debug then eval(valid_data) end
            if i % 200 == 0 then collectgarbage() end
            if i % opt.save_every_step == 0 then
                local savefile = string.format('%s_epoch%.2f_step%d_PPL%.2f.t7', opt.savefile, epoch, i, math.exp(train_loss/train_nonzeros))
                print('saving step checkpoint to ' .. savefile)
                clean_layer(generatorx)
                clean_layer(generatory)
            -- {encoder, decoder, generatorx, generatory, enc_embedx, enc_embedy, dec_embedx, dec_embedy, encoder_bwd}
                if opt.brnn == 0 then
                    torch.save(savefile, {{encoder, decoder, generatorx, generatory, enc_embedx, enc_embedy, dec_embedx, dec_embedy}, opt})
                else
                    torch.save(savefile, {{encoder, decoder, generatorx, generatory, enc_embedx, enc_embedy, dec_embedx, dec_embedy, encoder_bwd}, opt})
                end
            end
        end
        return train_loss, train_nonzeros
    end


    local total_loss, total_nonzeros, batch_loss, batch_nonzeros
    print('train running !!!')
    for epoch = opt.start_epoch, opt.epochs do
        generatorx:training() generatory:training()
        total_loss, total_nonzeros = train_batch(train_data, epoch)

        local train_score = math.exp(total_loss/total_nonzeros)
        print('Train', train_score)

        opt.train_perf[#opt.train_perf + 1] = train_score
        local score = eval(valid_data)
        opt.val_perf[#opt.val_perf + 1] = score
        if opt.optim == 'sgd' then decay_lr(epoch) end
        -- clean and save models
        local savefile = string.format('%s_epoch%.2f_%.2f.t7', opt.savefile, epoch, score)
        if epoch % opt.save_every == 0 then
            print('saving checkpoint to ' .. savefile)
            clean_layer(generatorx)
            clean_layer(generatory)
    -- {encoder, decoder, generatorx, generatory, enc_embedx, enc_embedy, dec_embedx, dec_embedy, encoder_bwd}
            if opt.brnn == 0 then
                torch.save(savefile, {{encoder, decoder, generatorx, generatory, enc_embedx, enc_embedy, dec_embedx, dec_embedy}, opt})
            else
                torch.save(savefile, {{encoder, decoder, generatorx, generatory, enc_embedx, enc_embedy, dec_embedx, dec_embedy, encoder_bwd}, opt})
            end
        end
    end
end


function eval(data)
    if opt.debug then print('eval begin') end
    encoder_clones[1]:evaluate()
    encoder_clones[1]:evaluate()
    generatorx:evaluate()
    generatory:evaluate()
    enc_embedx:evaluate()
    enc_embedy:evaluate()
    dec_embedx:evaluate()
    dec_embedy:evaluate()

    if opt.brnn == 1 then
        encoder_bwd_clones[1]:evaluate()
    end

    local nll = 0
    local total = 0
    for i = 1, data:size() do
        local d = data[i]
        local batch_l, target_l, source_l = d[5], d[6], d[7]
        local target, target_out, nonzeros_filter, source = d[1], d[2], d[3], d[4]
        target = target_out local nonzeros = torch.sum(nonzeros_filter:clone():double()) -- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! very important !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        source_x = map(source, src_mapx, opt.gpuid)
        source_y = map(source, src_mapy, opt.gpuid)
        target_x = map(target, trg_mapx, opt.gpuid2)
        target_y = map(target, trg_mapy, opt.gpuid2)
        setGPU(opt.gpuid)

        local rnn_state_enc = reset_state(init_fwd_enc, batch_l)
        local context       = context_proto[{{1, batch_l}, {1, source_l * 2}}]
        -- forward prop encoder
        if opt.debug then print('forward prop encoder') end
        for t = 1, source_l do
            local source_embedx = enc_embedx:forward(source_x[t])
            local encoder_input_x = {source_embedx, table.unpack(rnn_state_enc)}
            local out = encoder_clones[1]:forward(encoder_input_x)
            rnn_state_enc = out
            context[{{}, 2 * t - 1}]:copy(out[#out])

            local source_embedy = enc_embedy:forward(source_y[t])
            local encoder_input_y = {source_embedy, table.unpack(rnn_state_enc)}
            local out = encoder_clones[1]:forward(encoder_input_y)
            rnn_state_enc = out
            context[{{}, 2 * t}]:copy(out[#out])
        end
        local rnn_state_enc_bwd 
        if opt.brnn == 1 then
            rnn_state_enc_bwd = reset_state(init_fwd_enc, batch_l)
            for t = source_l, 1, -1 do
                local source_embedy = enc_embedy:forward(source_y[t])
                local encoder_input_y = {source_embedy, table.unpack(rnn_state_enc_bwd)}
                local out = encoder_bwd_clones[1]:forward(encoder_input_y)
                rnn_state_enc_bwd = out
                context[{{}, 2 * t}]:add(out[#out])

                -- second forward embedding x
                local source_embedx = enc_embedx:forward(source_x[t])
                local encoder_input_x = {source_embedx, table.unpack(rnn_state_enc_bwd)}
                local out = encoder_bwd_clones[1]:forward(encoder_input_x)
                rnn_state_enc_bwd = out
                context[{{}, 2 * t - 1}]:add(out[#out])
            end
        end

        -- go to gpu2
        if opt.debug then print('go to gpu2') end
        if flag.twoGPU then
            setGPU(opt.gpuid2)
            local context2 = context_proto2[{{1, batch_l}, {1, source_l * 2}}]
            context2:copy(context)
            context = context2
        end
        local rnn_state_dec = reset_state(init_fwd_dec, batch_l)
        for L = 1, opt.num_layers do
            rnn_state_dec[L * 2 - 1]:copy(rnn_state_enc[L * 2 - 1])
            rnn_state_dec[L * 2    ]:copy(rnn_state_enc[L * 2    ])
        end
        if opt.brnn == 1 then
            local tmp_state = reset_state(init_tmp2, batch_l)
            for L = 1, opt.num_layers do
                tmp_state[L * 2 - 1]:copy(rnn_state_enc_bwd[L * 2 - 1])
                tmp_state[L * 2    ]:copy(rnn_state_enc_bwd[L * 2    ])
            end
            for L = 1, opt.num_layers do
                --rnn_state_dec[L * 2 - 1]:add(rnn_state_enc_bwd[L * 2 - 1])
                --rnn_state_dec[L * 2    ]:add(rnn_state_enc_bwd[L * 2    ])
                rnn_state_dec[L * 2 - 1]:add(tmp_state[L * 2 - 1])
                rnn_state_dec[L * 2    ]:add(tmp_state[L * 2    ])
            end
        end

        if opt.debug then print('forward decoder to calculate loss') end
        local loss = 0
        local decoder_input_zero = torch.zeros(batch_l, opt.word_vec_size):cuda()
        for t = 1, target_l do
            local decoder_input_y, first, ctx
            if t == 1 then          first = decoder_input_zero
            else                    first = dec_embedy:forward(target_y[t - 1]) end
            if opt.attn == 1 then   ctx   = context
            else                    ctx   = context[{{}, source_l * 2}] end
            decoder_input_y = {first, ctx, table.unpack(rnn_state_dec)}
            local out = decoder_clones[1]:forward(decoder_input_y)
            rnn_state_dec = {}
            for j = 1, #out - 1 do table.insert(rnn_state_dec, out[j]) end
            local pred = generatorx:forward(out[#out])
            loss = loss + criterion:forward(pred, target_x[t])


            local decoder_input_x = {dec_embedx:forward(target_x[t]), ctx, table.unpack(rnn_state_dec)}
            local out = decoder_clones[1]:forward(decoder_input_x)
            rnn_state_dec = {}
            for j = 1, #out - 1 do table.insert(rnn_state_dec, out[j]) end
            local pred = generatory:forward(out[#out])
            loss = loss + criterion:forward(pred, target_y[t])
        end
        nll = nll + loss
        total = total + nonzeros
        -- {encoder, decoder, generatorx, generatory, enc_embedx, enc_embedy, dec_embedx, dec_embedy, encoder_bwd}
    end
    local valid = math.exp(nll/total)
    print('Valid', valid)
    collectgarbage()
    return valid
end


function main()
    opt = cmd:parse(arg)
    if opt.debug == 1 then opt.debug = true else opt.debug = false end
    if opt.gpuid > 0 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        if opt.gpuid2 > 0 then
            print('using CUDA on second GPU ' .. opt.gpuid2 .. '...')
        end
        require 'cutorch'
        require 'cunn'
        if opt.cudnn == 1 then
            print('loading cudnn...')
            require 'cudnn'
        end
        setGPU(opt.gpuid)
        cutorch.manualSeed(opt.seed)
    end

    -- Create the data loader class
    print('loading data ...')
    train_data = data.new(opt, opt.data_file)
    valid_data = data.new(opt, opt.val_data_file)

    print('done!')
    print(string.format('source vocab size: %d, target vocab size: %d',
    valid_data.source_size, valid_data.target_size))

    opt.max_sent_l_src = valid_data.source:size(2)
    opt.max_sent_l_targ = valid_data.target:size(2)
    opt.max_sent_l = math.max(opt.max_sent_l_src, opt.max_sent_l_targ)
    if opt.max_batch_l == '' then
        opt.max_batch_l = valid_data.batch_l:max()
    end

    print(string.format('source max sent len: %d, target max sent len: %d', 
    valid_data.source:size(2), valid_data.target:size(2)))

    -- Build models.lua
    if opt.train_from:len() == 0 then
        encoder = make_lstm(valid_data, opt, 'enc')
        decoder = make_lstm(valid_data, opt, 'dec')
        generatorx, generatory, criterion = make_generator(valid_data, opt)
        if opt.brnn == 1 then
            encoder_bwd = make_lstm(valid_data, opt, 'enc')
        end
        enc_embedx, enc_embedy, dec_embedx, dec_embedy = make_embed(valid_data, opt)
    else
        -- {encoder, decoder, generatorx, generatory, enc_embedx, enc_embedy, dec_embedx, dec_embedy, encoder_bwd}
        print('loading ' ..opt.train_from .. '...')
        local checkpoint = torch.load(opt.train_from)
        local model, model_opt = checkpoint[1], checkpoint[2]
        opt.num_layers = model_opt.num_layers
        opt.rnn_size   = model_opt.rnn_size
        opt.attn       = model_opt.attn
        opt.brnn       = model_opt.brnn
        opt.word_vec_size = model_opt.word_vec_size
        opt.src_mapx   = nil
        opt.trg_mapx   = nil

        encoder, decoder, generatorx, generatory, enc_embedx, enc_embedy, dec_embedx, dec_embedy = 
        model[1]:double(), model[2]:double(), model[3]:double(), model[4]:double(), model[5]:double(), model[6]:double(),
        model[7]:double(), model[8]:double()
        if opt.brnn == 1 then
            encoder_bwd = model[9]:double()
        end
        _, _, criterion = make_generator(valid_data, opt)
    end

    layers = {encoder, decoder, generatorx, generatory, enc_embedx, enc_embedy, dec_embedx, dec_embedy}
    layers_GPU = {opt.gpuid, opt.gpuid2, opt.gpuid2, opt.gpuid2, opt.gpuid, opt.gpuid, opt.gpuid2, opt.gpuid2}
    if opt.brnn == 1 then
        table.insert(layers, encoder_bwd)
        table.insert(layers_GPU, opt.gpuid)
    end

    if opt.optim ~= 'sgd' then
        layer_etas = {}
        optStates  = {}
        for i = 1, #layers do
            layer_etas[i] = opt.learning_rate
            optStates[i]  = {}
        end
    end

    if flag.oneGPU then
        for i = 1, #layers do
            if flag.twoGPU then setGPU(layers_GPU[i]) end
            layers[i]:cuda()
        end
        if flag.twoGPU then setGPU(opt.gpuid2) end
        criterion:cuda()
    end

    local src_vocab = valid_data.source_size
    local trg_vocab = valid_data.target_size
    local src_base  = math.ceil(math.sqrt(src_vocab))
    local trg_base  = math.ceil(math.sqrt(trg_vocab))
    print('src_base = ', src_base, 'trg_base = ', trg_base)
    print(opt)
    -- get src_mapx, src_mapy, trg_mapx, trg_mapy
    src_mapx = {} src_mapy = {} trg_mapx = {} trg_mapy = {}
    if opt.src_mapx ~= nil then
        src_mapx = opt.src_mapx
        src_mapy = opt.src_mapy
    elseif opt.src_mapxy:len() == 0 then
        print('randommapxy source')
        src_mapx, src_mapy = randommapxy(src_vocab, src_base)
    else
        src_mapx, src_mapy = mapxyfromfile(opt.src_mapxy, src_base)
    end

    if opt.trg_mapx ~= nil then
        trg_mapx = opt.trg_mapx
        trg_mapy = opt.trg_mapy
    elseif opt.trg_mapxy:len() == 0 then
        print('randommapxy target')
        trg_mapx, trg_mapy = randommapxy(trg_vocab, trg_base)
    else
        trg_mapx, trg_mapy = mapxyfromfile(opt.trg_mapxy, trg_base)
    end

    opt.src_mapx = src_mapx
    opt.src_mapy = src_mapy
    opt.trg_mapx = trg_mapx
    opt.trg_mapy = trg_mapy

    check_conflict(src_mapx, src_mapy, src_vocab, src_base)
    check_conflict(trg_mapx, trg_mapy, trg_vocab, trg_base)
    collectgarbage()
    train(train_data, valid_data)
end

main()
