BASE = 1000
function HASH(x, y) return x * BASE + y end
function UNHASH(v) return math.floor(v/BASE), v%BASE end

require 'nn'
require 'string'
require 'hdf5'
require 'nngraph'
require 'models.lua'
require 'data.lua'
require 'util.lua'

stringx = require('pl.stringx')
cmd = torch.CmdLine()

-- file location
cmd:option('-model', 'seq2seq_lstm_attn.t7', [[Path to model file]])
cmd:option('-src_file', '', [[Source to decode]])
cmd:option('-trg_file', '', [[True target]])
cmd:option('-output_file', 'pred.txt', [[Path to predicted sentences]])
cmd:option('-src_dict', 'data/demo.src.dict', [[Path to source vocab]])
cmd:option('-trg_dict', 'data/demo.targ.dict', [[Path to trg vocab]])

-- beam search options
cmd:option('-beam', 5, [[Beam size]])
cmd:option('-max_sent_l', 250, [[Maximum sent length]])
--cmd:option('-replace_unk', 0)
cmd:option('-gpuid', 1, [[-1 use cpu]])
cmd:option('-gpuid2', 2, [[-1 use cpu]])
cmd:option('-cudnn', 1, [[using 1 cudnn]])

opt = cmd:parse(arg)
if opt.gpuid2 == nil or opt.gpuid2 <= 0 then opt.gpuid2 = opt.gpuid end

function copy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in pairs(orig) do
            copy[orig_key] = orig_value
        end
    else
        copy = orig
    end
    return copy
end
function flat_to_rc(v, flat_index)
    local row = math.floor((flat_index - 1) / v:size(2)) + 1
    return row, (flat_index - 1) % v:size(2) + 1
end

function setGPU(id) 
    if id > 0 then cutorch.setDevice(id) end
end
local StateAll = torch.class("StateAll")
function StateAll.initial(start) return {start} end
function StateAll.advance(state, token)
    local new_state = copy(state)
    table.insert(new_state, token)
    return new_state
end
function StateAll.next(state) return state[#state] end

function wordidx2sent(sent, hash2word, skip_end)
    local t = {}
    local start_i, end_i
    skip_end = skip_end or true
    if skip_end then end_i = #sent - 2
    else end_i = #sent end
    for i = 2, end_i, 2 do
        local x, y = sent[i], sent[i + 1]
        local hs = HASH(x, y)
        table.insert(t, hash2word[hs])
    end
    return table.concat(t, ' ')
end

function idx2key(file)
    print('debug = ', file)
    local f = io.open(file, 'r')
    local t = {}
    for line in f:lines() do
        local c = {}
        for w in line:gmatch'([^%s]+)' do
            table.insert(c, w)
        end
        t[tonumber(c[2])] = c[1]
    end
    return t
end

function flip_table(u)
    local t = {}
    for key, value in pairs(u) do t[value] = key end
    return t
end
function hash2key(dic, mapx, mapy)
    local t = {}
    for id, key in pairs(dic) do
        local hs = HASH(mapx[id], mapy[id])
        t[hs] = key
    end
    return t
end
function clean_sent(sent)
    local s = stringx.replace(sent, UNK_WORD, '')
    s = stringx.replace(s, START_WORD, '')
    s = stringx.replace(s, END_WORD, '')
    return s
end
function sent2wordidx(sent, word2hash, src_or_trg)
    local t = {} local u = {}
    if src_or_trg == 0 then
        local hs = word2hash[START_WORD]
        local x, y = UNHASH(hs)
        table.insert(t, x) table.insert(t, y)
        table.insert(u, START_WORD)
    end
    for word in sent:gmatch'([^%s]+)' do
        local hs = word2hash[word] or word2hash[UNK_WORD]
        local x, y = UNHASH(hs)
        table.insert(t, x) table.insert(t, y)
        table.insert(u, word)
    end
    local hs = word2hash[END_WORD]
    local x, y = UNHASH(hs)
    table.insert(t, x) table.insert(t, y)
    table.insert(u, END_WORD)
    return torch.LongTensor(t), u
end

-- K is beam size
function generate_beam(model, initial, K, max_sent_l, source)
    encoder, decoder, generatorx, generatory, enc_embedx, enc_embedy, dec_embedx, dec_embedy = 
    model[1], model[2],model[3],   model[4],   model[5],   model[6],   model[7],   model[8]
    if model_opt.brnn == 1 then encoder_bwd = model[9] end
    -- reset decoder initial states
    if opt.gpuid > 0 and opt.gpuid2 > 0 then setGPU(opt.gpuid) end

    local n = max_sent_l * 2
    -- backpointer table
    local prev_ks = torch.LongTensor(n, K):fill(0)
    -- current states
    local next_ys = torch.LongTensor(n, K):fill(0)
    -- current scores
    local scores  = torch.FloatTensor(n, K)
    scores:zero()

    local source_l = math.min(source:size(1)/2, opt.max_sent_l) -- it is word's number
    local states   = {} -- store predicted word idx
    states[1]      = {}
    for k = 1, 1 do
        table.insert(states[1], initial)
        next_ys[1][k] = State.next(initial)
    end

    local source_input = source:view(source_l * 2, 1)

    local rnn_state_enc = {}
    for i = 1, #init_fwd_enc do
        table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
    end
    local context = context_proto[{{}, {1, source_l * 2}}]:clone() -- 1 x source_l*2 x rnn_size

    -- source_x[t] = source_input[2 * t - 1]
    -- source_y[t] = source_input[2 * t]
    for t = 1, source_l do
        local source_embedx = enc_embedx:forward(source_input[2 * t - 1])
        local encoder_input_x = {source_embedx, table.unpack(rnn_state_enc)}
        local out = encoder:forward(encoder_input_x)
        rnn_state_enc = out
        context[{{}, 2 * t - 1}]:copy(out[#out])

        local source_embedy = enc_embedy:forward(source_input[2 * t])
        local encoder_input_y = {source_embedy, table.unpack(rnn_state_enc)}
        local out = encoder:forward(encoder_input_y)
        rnn_state_enc = out
        context[{{}, 2 * t}]:copy(out[#out])
    end

    rnn_state_dec = {}
    rnn_state_dec2 = {}
    for i = 1, #init_fwd_dec do
        table.insert(rnn_state_dec, init_fwd_dec[i]:zero())
    end

    for L = 1, model_opt.num_layers do
        rnn_state_dec[L * 2 - 1]:copy(
        rnn_state_enc[L * 2 - 1]:expand(K, model_opt.rnn_size))
        rnn_state_dec[L * 2    ]:copy(
        rnn_state_enc[L * 2    ]:expand(K, model_opt.rnn_size))
    end

    if model_opt.brnn == 1 then
        for i = 1, #rnn_state_enc do
            rnn_state_enc[i]:zero()
        end
        for t = source_l, 1, -1 do
            local source_embedy = enc_embedy:forward(source_input[2 * t])
            local encoder_input_y = {source_embedy, table.unpack(rnn_state_enc)}
            local out = encoder_bwd:forward(encoder_input_y)
            rnn_state_enc = out
            context[{{}, 2 * t}]:add(out[#out])

            local source_embedx = enc_embedx:forward(source_input[2 * t - 1])
            local encoder_input_x = {source_embedx, table.unpack(rnn_state_enc)}
            local out = encoder_bwd:forward(encoder_input_x)
            rnn_state_enc = out
            context[{{}, 2 * t - 1}]:add(out[#out])
        end
        for L = 1, model_opt.num_layers do
            rnn_state_dec[L * 2 - 1]:add(
            rnn_state_enc[L * 2 - 1]:expand(K, model_opt.rnn_size))
            rnn_state_dec[L * 2    ]:add(
            rnn_state_enc[L * 2    ]:expand(K, model_opt.rnn_size))
        end
    end
    

    context = context:expand(K, source_l * 2, model_opt.rnn_size)
    if opt.gpuid > 0 and opt.gpuid2 > 0 then
        setGPU(opt.gpuid2)
        local context2 = context_proto2[{{1, K}, {1, source_l * 2}}]
        context2:copy(context)
        context = context2
        assert(#init_fwd_dec2 == 2 * model_opt.num_layers)
        for i = 1, #init_fwd_dec2 do 
            table.insert(rnn_state_dec2, init_fwd_dec2[i]:zero()) 
            rnn_state_dec2[i]:copy(rnn_state_dec[i])
        end
        rnn_state_dec = rnn_state_dec2
    end

    out_float = torch.FloatTensor()
    local i = 1
    local done = false
    local max_score = -1e9
    local found_eos = false
    local decoder_input_zero = torch.zeros(K, model_opt.word_vec_size):cuda()
    local stop_generate = {} stop_generate[1] = {}
    while (not done) and (i < n) do
        i = i + 1
        states[i] = {}
        stop_generate[i] = {}
        -- local next_ys = torch.LongTensor(2 * max_sent_l, K):fill(0)
        local decoder_input1, decoder_input, first, ctx, dec_embed, generator
        decoder_input1 = next_ys:narrow(1, i - 1, 1):squeeze()  -- K vector
        if opt.beam == 1 then decoder_input1 = torch.LongTensor({decoder_input1}) end--!!! transfer to CudaTensor 
        if i%2 == 0 then dec_embed = dec_embedy else dec_embed = dec_embedx end
        if i == 2   then first = decoder_input_zero else first = dec_embed:forward(decoder_input1) end
        if model_opt.attn == 1 then ctx = context
        else                        ctx = context[{{}, source_l * 2}] end
        if i%2 == 0 then generator = generatorx else generator = generatory end
        decoder_input = {first, ctx, table.unpack(rnn_state_dec)}

        local out_decoder = decoder:forward(decoder_input)
        local out         = generator:forward(out_decoder[#out_decoder]) -- K x vocab_size(sqrt_trg_vocab) 

        rnn_state_dec = {}  -- to be modified later 2*layer_number x K x rnn_size
        for j = 1, #out_decoder - 1 do
            table.insert(rnn_state_dec, out_decoder[j])
        end
        out_float:resize(out:size()):copy(out)

        for k = 1, K do
            --State.disallow(out_float:select(1, k)) -- we may not be good at this since pad consist of two
            out_float[k]:add(scores[i - 1][k])       
        end

        local flat_out = out_float:view(-1)
        if i == 2 then flat_out = out_float[1] end -- all outputs same for first batch

        for k = 1, K do
            while true do
                local score, index = flat_out:max(1)
                local score = score[1]
                local prev_k, y_i = flat_to_rc(out_float, index[1])
                if stop_generate[i - 1][prev_k] then
                    flat_out[index[1]] = -1e9
                else
                    states[i][k] = State.advance(states[i - 1][prev_k], y_i)
                    prev_ks[i][k] = prev_k
                    next_ys[i][k] = y_i
                    scores[i][k]  = score
                    flat_out[index[1]] = -1e9
                    break
                end
            end
        end
        for j = 1, #rnn_state_dec do
            rnn_state_dec[j]:copy(rnn_state_dec[j]:index(1, prev_ks[i]))
        end
        -- check end
        if i%2 == 1 then -- two consists of one word
            end_hyp = states[i][1]
            end_score = scores[i][1]
            for k = 1, K do
                local possible_sent = states[i][k]
                local hs = HASH(possible_sent[#possible_sent - 1], possible_sent[#possible_sent])
                local lw = hash2word_trg[hs]
                if lw == END_WORD or lw == PAD_WORD then 
                    found_eos = true
                    if k == 1 then done = true end
                    if scores[i][k] > max_score then
                        max_hyp = possible_sent
                        max_score = scores[i][k]
                    end
                end
                if lw == END_WORD or lw == PAD_WORD then
                    stop_generate[i][k] = true
                end
            end
        end
    end
    if max_hyp == nil or end_score > max_score or not found_eos then
        max_hyp = end_hyp
        max_score = end_score
    end
    return max_hyp, max_score, states[i], scores[i]
end

function main()
    PAD = 1; UNK = 2; START = 3; END = 4
    PAD_WORD = '<blank>'; UNK_WORD = '<unk>'; START_WORD = '<s>'; END_WORD = '</s>'
    MAX_SENT_L = opt.max_sent_l
    assert(path.exists(opt.src_file), 'src_file no')
    assert(path.exists(opt.model), 'model no')

    -- parse input params
    opt = cmd:parse(arg)
    if opt.gpuid >= 0 then
        require 'cutorch'
        require 'cunn'
        if opt.cudnn == 1 then
            require 'cudnn'
        end
    end
    print('loading ' .. opt.model .. '...')
    checkpoint = torch.load(opt.model)
    print('model load done!')

    -- we don't do after replacement

    -- load model and word2idx/idx2word dictionaries
    model, model_opt = checkpoint[1], checkpoint[2]
    model_opt.brnn = model_opt.brnn or 1
    model_opt.attn = model_opt.attn or 1
    assert(model_opt.src_mapx ~= nil, 'mapx and mapy all ready')

    idx2word_src = idx2key(opt.src_dict)
    word2idx_src = flip_table(idx2word_src)
    idx2word_trg = idx2key(opt.trg_dict)
    word2idx_trg = flip_table(idx2word_trg)

    print('src_word number = ', #idx2word_src, 'trg_word number = ', #idx2word_trg)
    print('get hash2word_src hash2word_trg')
    hash2word_src = hash2key(idx2word_src, model_opt.src_mapx, model_opt.src_mapy)
    hash2word_trg = hash2key(idx2word_trg, model_opt.trg_mapx, model_opt.trg_mapy)
    word2hash_src = flip_table(hash2word_src)
    word2hash_trg = flip_table(hash2word_trg)

    -- load gold labels if it exists, we first not consider this
    opt.score_gold = 0
    if path.exists(opt.trg_file) then
    end

    -- model transfer
    -- we need a mapping 
    layers_GPU = {opt.gpuid, opt.gpuid2, opt.gpuid2, opt.gpuid2, opt.gpuid, opt.gpuid, opt.gpuid2, opt.gpuid2}
    if model_opt.brnn == 1 then
        table.insert(layers_GPU, opt.gpuid)
    end

    print('set models parts into gpus')
    print(layers_GPU)
    for i = 1, #model do
        setGPU(layers_GPU[i])
        model[i]:double():cuda()
        model[i]:evaluate()
    end

    context_proto = torch.zeros(1, MAX_SENT_L * 2, model_opt.rnn_size)
    local h_init_dec = torch.zeros(opt.beam, model_opt.rnn_size)
    local h_init_dec2 = torch.zeros(opt.beam, model_opt.rnn_size)
    local h_init_enc = torch.zeros(1, model_opt.rnn_size)
    if opt.gpuid > 0 then
        setGPU(opt.gpuid)
        h_init_enc = h_init_enc:cuda()
        h_init_dec = h_init_dec:cuda()
        if opt.gpuid2 > 0 then
            setGPU(opt.gpuid)
            context_proto = context_proto:cuda()
            setGPU(opt.gpuid2)
            context_proto2 = torch.zeros(opt.beam, MAX_SENT_L * 2, model_opt.rnn_size):cuda()
            h_init_dec2    = h_init_dec2:cuda()
        else
            context_proto  = context_proto:cuda()
        end
    end

    setGPU(opt.gpuid)
    init_fwd_enc = {}
    init_fwd_dec = {}
    init_fwd_dec2 = {}

    for L = 1, model_opt.num_layers do
        table.insert(init_fwd_enc, h_init_enc:clone())
        table.insert(init_fwd_enc, h_init_enc:clone())
        table.insert(init_fwd_dec, h_init_dec:clone()) -- cell
        table.insert(init_fwd_dec, h_init_dec:clone()) -- hidden
    end

    setGPU(opt.gpuid2)
    for L = 1, model_opt.num_layers do
        table.insert(init_fwd_dec2, h_init_dec2:clone()) -- cell
        table.insert(init_fwd_dec2, h_init_dec2:clone()) -- hidden
    end
    setGPU(opt.gpuid)

    pred_score_total = 0
    pred_words_total = 0

    State = StateAll
    local sent_id = 0
    pred_sents = {}
    local file = io.open(opt.src_file, 'r')
    local out_file = io.open(opt.output_file, 'w')
    print('open file ', opt.output_file, ' deal with sentences')
    for line in file:lines() do
        sent_id = sent_id + 1
        line = clean_sent(line)
        print('SENT ' .. sent_id .. ': ' .. line)
        local source, source_str = sent2wordidx(line, word2hash_src, 0)
        -------------------------------------------------------------------------------------------------------------------
        state = State.initial(0) -- seems not right !!!! modify it, I first assume a wrong number 0, to infer embedding all zero vector
        pred, pred_score, all_sents, all_scores = generate_beam(model, state, opt.beam, MAX_SENT_L, source)
        pred_score_total = pred_score_total + pred_score
        pred_words_total = pred_words_total + (#pred - 1)/2 - 1
        pred_sent = wordidx2sent(pred, hash2word_trg, true)
        -------------------------------------------------------------------------------------------------------------------
        out_file:write(pred_sent .. '\n')
        -- we first not consider n_best
        print(pred_sent .. '\n')
        print('')
    end
    print(string.format('PRED AVG SCORE: %.4f, PRED PPL: %.4f', pred_score_total / pred_words_total,
    math.exp(-pred_score_total/pred_words_total)))
    out_file:close()
end

main()

