import numpy as np
import bchlib
import binascii
import math
from pydtmc import MarkovChain
from scipy.stats import bernoulli as bern
from hmmlearn import hmm
import huffman

#Require this function for calling functions from "bchlib": they require data in byte format
def binary_string_to_bytes(bin_string):
    '''
    Convert binary string to bytes (needed for BCH).
    Throw error if bin_string length not multiple of 8.
    '''
    if len(bin_string) % 8 != 0:
        raise Exception('binary string length not multiple of 8')
    return binascii.unhexlify((((hex(int(bin_string, 2)))[2:]).rstrip('L')).zfill(len(bin_string)//4))

#Require this function for calling functions from "bchlib": converting output from byte format to string
def bytes_to_binary_string(byte_string):
    '''
    Convert bytes to binary string (needed for BCH).
    '''
    return bin(int(binascii.hexlify(byte_string), 16))[2:].zfill(len(byte_string)*8)

#S=Number of States
#M=Number of Message possibilities
#n=Packet Length
#BCH_Poly=37 
#Bits_per_error=5
#samples=Number of packets used for training
#p=Bit flip probability of BSC
#trans_mat=Transition Matrix of Markov Source
#test_size=Number of packets used for testing

#Function for learning p and the transition matrix, sends packets with only state info (not included in the paper)
def training(S,n,BCH_Poly,Bits_per_error,samples,p,trans_mat):

    binary_length_S=math.ceil(math.log2(S))
    BCH_t=int((n-binary_length_S)/Bits_per_error)
    packet_pseudo_size=binary_length_S+BCH_t*Bits_per_error
    mc=MarkovChain(trans_mat)
    bch=bchlib.BCH(BCH_Poly,BCH_t)

    states=mc.walk(samples)
    states=np.asarray(states)
    states=states.astype(int)-1

    net_bitflips=0
    trans_mat_predicted=np.zeros((S,S))
    predicted_states=np.empty(samples)

    #For first packet
    #Encoding
    state_in_binary=format(states[0],'b').zfill(binary_length_S)
    data=binary_string_to_bytes(state_in_binary.zfill(8))
    ecc=bch.encode(data)
    parity=bytes_to_binary_string(ecc)
    parity=np.array(list(parity), dtype=int)
    parity_len=BCH_t*Bits_per_error
    bits_removed=parity.size-parity_len
    parity=parity[0:parity_len]
    state_in_binary=np.array(list(state_in_binary), dtype=int)
    packet=np.append(state_in_binary,parity)
    #Transmission through channel
    r=bern.rvs(p,size=packet_pseudo_size)
    packet_received=(packet+r)%2
    #Decoding
    data_received=packet_received[0:binary_length_S]
    ecc_received=packet_received[binary_length_S:]
    data_received="".join(str(e) for e in data_received)
    ecc_received="".join(str(e) for e in ecc_received)
    data_received=binary_string_to_bytes(data_received.zfill(8))
    ecc_received=binary_string_to_bytes(ecc_received.ljust(bits_removed+parity_len,'0'))
    bitflips,correct_data,correct_ecc=bch.decode(data_received,ecc_received)
    correct_data=bytes_to_binary_string(correct_data)
    correct_data=np.array(list(correct_data),dtype=int)[-binary_length_S:]
    predicted_state=int("".join(str(x) for x in correct_data),2)
    if(bitflips==-1):
        predicted_states[0]=-1
        prev_state=-1
        net_bitflips+=5
    else:
        predicted_states[0]=predicted_state
        prev_state=predicted_state
        net_bitflips+=bitflips
    prev_bitflips=bitflips

    #For rest of packets
    for i in range(1,samples):
        #Encoding
        state_in_binary=format(states[i],'b').zfill(binary_length_S)
        data=binary_string_to_bytes(state_in_binary.zfill(8))
        ecc=bch.encode(data)
        parity=bytes_to_binary_string(ecc)
        parity=np.array(list(parity), dtype=int)
        parity_len=BCH_t*Bits_per_error
        bits_removed=parity.size-parity_len
        parity=parity[0:parity_len]
        state_in_binary=np.array(list(state_in_binary), dtype=int)
        packet=np.append(state_in_binary,parity)
        #Transmission
        r=bern.rvs(p,size=packet_pseudo_size)
        packet_received=(packet+r)%2
        #Decoding
        data_received=packet_received[0:binary_length_S]
        ecc_received=packet_received[binary_length_S:]
        data_received="".join(str(e) for e in data_received)
        ecc_received="".join(str(e) for e in ecc_received)
        data_received=binary_string_to_bytes(data_received.zfill(8))
        ecc_received=binary_string_to_bytes(ecc_received.ljust(bits_removed+parity_len,'0'))
        bitflips,correct_data,correct_ecc=bch.decode(data_received,ecc_received)
        correct_data=bytes_to_binary_string(correct_data)
        correct_data=np.array(list(correct_data),dtype=int)[-binary_length_S:]
        predicted_state=int("".join(str(x) for x in correct_data),2)
        if(bitflips!=-1 and prev_bitflips!=-1):
            trans_mat_predicted[prev_state][predicted_state]+=1
            prev_state=predicted_state
            predicted_states[i]=predicted_state
            net_bitflips+=bitflips
        elif(bitflips==-1):
            prev_state=-1
            predicted_states[i]=-1
            net_bitflips+=5
        elif(bitflips!=-1):
            predicted_states[i]=predicted_state
        prev_bitflips=bitflips

    trans_mat_predicted=trans_mat_predicted/np.sum(trans_mat_predicted,axis=1)[:,None]
    p_predicted=net_bitflips/(samples*packet_pseudo_size)

    return p_predicted, trans_mat_predicted

#Generates a matrix with the code for legacy encoding scheme (refer paper)
def code_combined(S,M,n,BCH_Poly,Bits_per_error):
    binary_length_S=math.ceil(math.log2(S))
    binary_length_M=math.ceil(math.log2(M))
    BCH_t=math.ceil((n-binary_length_S-binary_length_M)/Bits_per_error)
    parity_len=BCH_t*Bits_per_error
    bch=bchlib.BCH(BCH_Poly,BCH_t)
    codes_temp=np.empty((S*M,binary_length_S+binary_length_M+parity_len),dtype=int)
    for x in range(0,S):
        for y in range(0,M):
            x_in_binary=format(x,'b').zfill(binary_length_S)
            y_in_binary=format(y,'b').zfill(binary_length_M)
            x_in_binary=format(x,'b').zfill(binary_length_S)
            y_in_binary=format(y,'b').zfill(binary_length_M)
            xy_in_binary=x_in_binary+y_in_binary
            xy_in_bytes=binary_string_to_bytes(xy_in_binary.zfill(16))
            ecc=bch.encode(xy_in_bytes)
            parity=bytes_to_binary_string(ecc)
            parity=np.array(list(parity), dtype=int)[0:parity_len]
            xy_in_binary=np.array(list(xy_in_binary), dtype=int)
            tot=np.append(xy_in_binary,parity)
            codes_temp[x*M+y,:]=tot

    codes=codes_temp[:,binary_length_S+binary_length_M+parity_len-n:binary_length_S+binary_length_M+parity_len]
    #codes=codes_temp[:,0:n]
    return codes

#Generates a matrix with the code for punctured encoding scheme (our proposed scheme, refer paper)
def code_punctured(S,M,n,BCH_Poly,Bits_per_error):
    binary_length_S=math.ceil(math.log2(S))
    binary_length_M=math.ceil(math.log2(M))
    BCH_t=math.ceil((n-binary_length_S-binary_length_M)/Bits_per_error)+1
    parity_len=BCH_t*Bits_per_error
    bch=bchlib.BCH(BCH_Poly,BCH_t)
    codes_temp=np.empty((S*M,binary_length_S+binary_length_M+parity_len),dtype=int)
    for x in range(0,S):
        for y in range(0,M):
            x_in_binary=format(x,'b').zfill(binary_length_S)
            y_in_binary=format(y,'b').zfill(binary_length_M)
            x_in_binary=format(x,'b').zfill(binary_length_S)
            y_in_binary=format(y,'b').zfill(binary_length_M)
            xy_in_binary=x_in_binary+y_in_binary
            xy_in_bytes=binary_string_to_bytes(xy_in_binary.zfill(16))
            ecc=bch.encode(xy_in_bytes)
            parity=bytes_to_binary_string(ecc)
            parity=np.array(list(parity), dtype=int)[0:parity_len]
            xy_in_binary=np.array(list(xy_in_binary), dtype=int)
            tot=np.append(xy_in_binary,parity)
            codes_temp[x*M+y,:]=tot

    codes=np.delete(codes_temp,np.s_[0:binary_length_S+binary_length_M+parity_len-n],axis=1)
    return codes

#Generates a matrix with the code for nested encoding scheme (not included in paper)
#First apply BCH on message part and then concatenate state with encoded message and then again apply BCH
#binary_length_encoded_message=Length of encoded message
def code_nested(S,M,n,BCH_Poly,Bits_per_error,binary_length_encoded_message):
    binary_length_S=math.ceil(math.log2(S))
    binary_length_M=math.ceil(math.log2(M))
    BCH_t_inner=int((binary_length_encoded_message-binary_length_M)/Bits_per_error)
    BCH_t_outer=int((n-binary_length_S-binary_length_encoded_message)/Bits_per_error)
    bch_inner=bchlib.BCH(BCH_Poly,BCH_t_inner)
    bch_outer=bchlib.BCH(BCH_Poly,BCH_t_outer)
    codes=np.empty((S*M,n),dtype=int)
    parity_len_outer=BCH_t_outer*Bits_per_error
    parity_len_inner=BCH_t_inner*Bits_per_error
    for x in range(0,S):
        for y in range(0,M):
            x_in_binary=format(x,'b').zfill(binary_length_S)
            y_in_binary=format(y,'b').zfill(binary_length_M)

            y_encoding=bch_inner.encode(binary_string_to_bytes(y_in_binary.zfill(8)))
            y_encoding_in_binary=bytes_to_binary_string(y_encoding)
            y_encoding_in_binary=y_encoding_in_binary[0:parity_len_inner]

            xy_in_binary=x_in_binary+y_in_binary+y_encoding_in_binary
            xy_in_bytes=binary_string_to_bytes(xy_in_binary.zfill(16))
            ecc=bch_outer.encode(xy_in_bytes)
            parity=bytes_to_binary_string(ecc)
            parity=np.array(list(parity), dtype=int)[0:parity_len_outer]
            xy_in_binary=np.array(list(xy_in_binary), dtype=int)
            tot=np.append(xy_in_binary,parity)
            codes[x*M+y,:]=tot

    return codes

#Generates two matrix with codes for separated encoding scheme (not included in paper)
#Applies BCH on state and message separately, overall code is concatenation of the encoded state and encoded message
#n_state=Number of bits in packet alloted to encoded state
#n_message=Number of bits in packet alloted to encoded message
#n_state+n_message=n
#Returns code matrices for state and message separately
def code_separate(S,M,n_state,n_message,BCH_Poly,Bits_per_error):
    binary_length_S=math.ceil(math.log2(S))
    binary_length_M=math.ceil(math.log2(M))
    BCH_t_state=math.ceil((n_state-binary_length_S)/Bits_per_error)
    BCH_t_message=math.ceil((n_message-binary_length_M)/Bits_per_error)
    bch_state=bchlib.BCH(BCH_Poly,BCH_t_state)
    bch_message=bchlib.BCH(BCH_Poly,BCH_t_message)
    parity_len_state=BCH_t_state*Bits_per_error
    parity_len_message=BCH_t_message*Bits_per_error
    codes_state=np.empty((S,binary_length_S+parity_len_state),dtype=int)
    codes_message=np.empty((M,binary_length_M+parity_len_message),dtype=int)
    for x in range(0,S):
        x_in_binary=format(x,'b').zfill(binary_length_S)
        x_in_bytes=binary_string_to_bytes(x_in_binary.zfill(8))
        ecc=bch_state.encode(x_in_bytes)
        parity=bytes_to_binary_string(ecc)
        parity=np.array(list(parity), dtype=int)[0:parity_len_state]
        x_in_binary=np.array(list(x_in_binary), dtype=int)
        tot=np.append(x_in_binary,parity)
        codes_state[x,:]=tot

    for y in range(0,M):
        y_in_binary=format(y,'b').zfill(binary_length_M)
        y_in_bytes=binary_string_to_bytes(y_in_binary.zfill(8))
        ecc=bch_message.encode(y_in_bytes)
        parity=bytes_to_binary_string(ecc)
        parity=np.array(list(parity), dtype=int)[0:parity_len_message]
        y_in_binary=np.array(list(y_in_binary), dtype=int)
        tot=np.append(y_in_binary,parity)
        codes_message[y,:]=tot

    codes_state=codes_state[:,binary_length_S+parity_len_state-n_state:binary_length_S+parity_len_state]
    codes_message=codes_message[:,binary_length_M+parity_len_message-n_message:binary_length_M+parity_len_message]
    return codes_state,codes_message

#Generates a matrix with the code for stationary compression encoding scheme (refer paper)
def code_comp_stat(S,M,n,BCH_Poly,Bits_per_error,transmat):
    mc=MarkovChain(transmat)
    stat_dist=(mc.steady_states)[0]
    symbols=[]
    for i in range(0,S):
        symbols.append((i,stat_dist[i]))

    huffman_code=(huffman.codebook(symbols))

    binary_length_M=math.ceil(math.log2(M))
    codes_comp=np.empty((S*M,n),dtype=int)
    for x in range(0,S):
        for y in range(0,M):
            binary_length_S=len(huffman_code[x])
            BCH_t=math.ceil((n-binary_length_S-binary_length_M)/Bits_per_error)
            parity_len=BCH_t*Bits_per_error
            bch=bchlib.BCH(BCH_Poly,BCH_t)
            y_in_binary=format(y,'b').zfill(binary_length_M)
            xy_in_binary=huffman_code[x]+y_in_binary
            xy_in_bytes=binary_string_to_bytes(xy_in_binary.zfill(16))
            ecc=bch.encode(xy_in_bytes)
            parity=bytes_to_binary_string(ecc)
            parity=np.array(list(parity), dtype=int)[0:parity_len]
            xy_in_binary=np.array(list(xy_in_binary), dtype=int)
            tot=np.append(xy_in_binary,parity)
            codes_comp[x*M+y,:]=tot[binary_length_S+binary_length_M+parity_len-n:binary_length_S+binary_length_M+parity_len]
            #codes_comp[x*M+y,:]=tot[0:n]
            #codes_comp[x*M+y,:]=np.append(tot[0:binary_length_S+binary_length_M],tot[2*binary_length_S+2*binary_length_M+parity_len-n:binary_length_S+binary_length_M+parity_len])
            
    return codes_comp

#Function to generate states randomly based on the Markov chain and messags uniformly
def data_generator(M,trans_mat,test_size):
    messages=np.empty(test_size,dtype=int)
    mc=MarkovChain(trans_mat)
    states=mc.walk(test_size)
    states=np.asarray(states)
    states=states.astype(int)-1

    for i in range(0,test_size):
        message=np.random.randint(M)
        messages[i]=message

    return states,messages

#Takes states and messages from data_generator and the code and outputs a list of packets
def encoder(M,states,messages,codes,test_size):
    sent=[]
    for i in range(0,test_size):
        sent.append(codes[states[i]*M+messages[i],:])
    return sent

def encoder_separate(states,messages,codes_state,codes_message,test_size):
    sent=[]
    for i in range(0,test_size):
        packet=np.append(codes_state[states[i]],codes_message[messages[i]])
        sent.append(packet)
    return sent

#BSC channel: Each bit is flipped with probability p
def channel(sent,test_size,p):
    received=[]
    a=sent[0]
    packet_pseudo_size=a.size
    for i in range(0,test_size):
        r=bern.rvs(p,size=packet_pseudo_size)
        received.append((sent[i]+r)%2)

    return received

#Decoder: Takes list of recieved packets from channel and decodes based on the decoding_type and delay
#delay: Number of packets from the future used for prediction
#decoding_types:
#HMM_no_delay_Viterbi: Use Viterbi Algorithm for HMM decoding without using any future packets (only use present and past packets)
#BCH:Use MLE decoding (i.e.only the present packet, closest code from the set of codes)
#HMM_no_delay_MAP: Use Forward Backward algorithm for HMM decoding without using any future packets (MAP based on present and past packets)
#HMM_delay_Viterbi:Use Viterbi Algorithm for HMM decoding using "delay" number of packets from the future
#HMM_delay_Viterbi:Use Forward backward Algorithm for HMM decoding using "delay" number of packets from the future
def decoder(S,M,n,received,trans_mat_predicted,p_predicted,codes,test_size,decoding_type,delay):
    state_predicted=np.empty(test_size,dtype=int)
    message_predicted=np.empty(test_size,dtype=int)

    if(decoding_type=="HMM_no_delay_Viterbi"):

        startprob=np.full(S,1/S)
        message_number=np.empty((S,test_size),dtype=int)
        T1=np.empty(S)
        prev_T1=np.empty(S)
        B=np.empty(S)

        #For first packet
        packet=received[0]
        compare=(codes!=packet)
        hammdist=np.sum(compare,axis=1)
        for l in range(0,S):
            index=np.argmin(hammdist[M*l:M*l+M])
            message_number[l,0]=index
            B[l]=np.sum(np.power(p_predicted,hammdist[M*l:M*l+M])*np.power((1-p_predicted),n-hammdist[M*l:M*l+M]))
        T1=startprob*B

        state_predicted[0]=np.argmax(T1)
        message_predicted[0]=message_number[state_predicted[0],0]

        for j in range(1,test_size):
            prev_T1=np.array(T1)
            packet=received[j]
            compare=(codes!=packet)
            hammdist=np.sum(compare,axis=1)
            for l in range(0,S):
                index=np.argmin(hammdist[M*l:M*l+M])
                message_number[l,j]=index
                B[l]=np.sum(np.power(p_predicted,hammdist[M*l:M*l+M])*np.power((1-p_predicted),n-hammdist[M*l:M*l+M]))
                index_max_apriori=np.argmax(prev_T1*trans_mat_predicted[:,l])
                T1[l]=prev_T1[index_max_apriori]*trans_mat_predicted[index_max_apriori,l]*B[l]
            if(T1[0]<10**(-100)):
                T1=T1*10**100

            state_predicted[j]=np.argmax(T1)
            message_predicted[j]=message_number[state_predicted[j],j]

        return state_predicted,message_predicted


    if(decoding_type=="BCH"):
        for j in range(0,test_size):
            packet=received[j]
            compare=(codes!=packet)
            hammdist=np.sum(compare,axis=1)
            index_min_dist=np.argmin(hammdist)
            state_predicted[j]=int(index_min_dist/M)
            message_predicted[j]=index_min_dist%M

        return state_predicted,message_predicted

    if(decoding_type=="HMM_no_delay_MAP"):

        startprob=np.full(S,1/S)
        message_number=np.empty((S,test_size),dtype=int)
        T1=np.empty(S)
        prev_T1=np.empty(S)
        B=np.empty(S)

        #For first packet
        packet=received[0]
        compare=(codes!=packet)
        hammdist=np.sum(compare,axis=1)
        for l in range(0,S):
            index=np.argmin(hammdist[M*l:M*l+M])
            message_number[l,0]=index
            B[l]=np.sum(np.power(p_predicted,hammdist[M*l:M*l+M])*np.power((1-p_predicted),n-hammdist[M*l:M*l+M]))
        T1=startprob*B

        state_predicted[0]=np.argmax(T1)
        message_predicted[0]=message_number[state_predicted[0],0]

        for j in range(1,test_size):
            prev_T1=np.array(T1)
            packet=received[j]
            compare=(codes!=packet)
            hammdist=np.sum(compare,axis=1)
            for l in range(0,S):
                index=np.argmin(hammdist[M*l:M*l+M])
                message_number[l,j]=index
                B[l]=np.sum(np.power(p_predicted,hammdist[M*l:M*l+M])*np.power((1-p_predicted),n-hammdist[M*l:M*l+M]))
                apriori=np.sum(prev_T1*trans_mat_predicted[:,l])
                T1[l]=apriori*B[l]
            if(T1[0]<10**(-100)):
                T1=T1*10**100

            state_predicted[j]=np.argmax(T1)
            message_predicted[j]=message_number[state_predicted[j],j]

        return state_predicted,message_predicted


    if(decoding_type=="HMM_delay_Viterbi"):
        startprob=np.full(S,1/S)
        message_number=np.empty((S,test_size),dtype=int)
        T1=np.empty((S,test_size))
        Z=np.empty(test_size,dtype=int)
        T2=np.empty((S,test_size),dtype=int)
        B=np.empty(S)

        packet=received[0]
        compare=(codes!=packet)
        hammdist=np.sum(compare,axis=1)
        for l in range(0,S):
            index=np.argmin(hammdist[M*l:M*l+M])
            message_number[l,0]=index
            B[l]=np.sum(np.power(p_predicted,hammdist[M*l:M*l+M])*np.power((1-p_predicted),n-hammdist[M*l:M*l+M]))

        T1[:,0]=startprob*B
        T2[:,0]=0

        for j in range(1,test_size):
            packet=received[j]
            compare=(codes!=packet)
            hammdist=np.sum(compare,axis=1)
            for l in range(0,S):
                index=np.argmin(hammdist[S*l:S*l+S])
                message_number[l,j]=index
                B[l]=np.sum(np.power(p_predicted,hammdist[S*l:S*l+S])*np.power((1-p_predicted),n-hammdist[S*l:S*l+S]))

            for i in range(0,S):
                T2[i,j]=np.argmax(T1[:,j-1]*trans_mat_predicted[:,i])
                T1[i,j]=T1[T2[i,j],j-1]*trans_mat_predicted[T2[i,j],i]*B[i]

            if(T1[0,j]<10**(-100)):
                T1[:,j]=T1[:,j]*10**100

            Z[j]=np.argmax(T1[:,j])
            if(j>=delay):
                k=j
                while (k>(j-delay)):
                    Z[k-1]=T2[Z[k],k]
                    k-=1
                state_predicted[j-delay]=Z[j-delay]


        state_predicted[test_size-delay:test_size]=Z[test_size-delay:test_size]

        for i in range(0,test_size):
            message_predicted[i]=message_number[state_predicted[i],i]

        return state_predicted,message_predicted


    if(decoding_type=="HMM_delay_MAP"):
        startprob=np.full(S,1/S)
        message_number=np.empty((S,test_size),dtype=int)
        forward=np.empty((S,test_size))
        backward=np.zeros((S,delay+1))
        B=np.empty((S,test_size))
        alpha=np.empty(S)

        packet=received[0]
        compare=(codes!=packet)
        hammdist=np.sum(compare,axis=1)
        for l in range(0,S):
            index=np.argmin(hammdist[M*l:M*l+M])
            message_number[l,0]=index
            B[l,0]=np.sum(np.power(p_predicted,hammdist[M*l:M*l+M])*np.power((1-p_predicted),n-hammdist[M*l:M*l+M]))

        forward[:,0]=startprob*B[:,0]

        for j in range(1,test_size):
            backward.fill(0)
            packet=received[j]
            compare=(codes!=packet)
            hammdist=np.sum(compare,axis=1)
            for l in range(0,S):
                index=np.argmin(hammdist[S*l:S*l+S])
                message_number[l,j]=index
                B[l,j]=np.sum(np.power(p_predicted,hammdist[S*l:S*l+S])*np.power((1-p_predicted),n-hammdist[S*l:S*l+S]))

            for i in range(0,S):
                apriori=np.sum(forward[:,j-1]*trans_mat_predicted[:,i])
                forward[i,j]=apriori*B[i,j]

            if(forward[0,j]<10**(-100)):
                forward[:,j]=forward[:,j]*10**100

            backward[:,delay].fill(1)
            for k in range(delay-1,-1,-1):
                for i in range(0,S):
                    alpha=np.squeeze(np.asarray(backward[:,k+1]))*np.squeeze(np.asarray(trans_mat_predicted[i,:]))
                    backward[i,k]=np.sum(alpha*B[:,j+k-delay+1])

            state_predicted[j-delay]=np.argmax(forward[:,j-delay]*backward[:,0])


        for i in range(test_size-delay,test_size):
            state_predicted[i]=np.argmax(forward[:,i]*backward[:,delay+i-test_size])

        for i in range(0,test_size):
            message_predicted[i]=message_number[state_predicted[i],i]

        return state_predicted,message_predicted

    
    
#Decoder for the separate encoding scheme
def decoder_separate(S,M,n_state,n_message,received,trans_mat_predicted,p_predicted,codes_state,codes_message,test_size,decoding_type,delay):
    state_predicted=np.empty(test_size,dtype=int)
    message_predicted=np.empty(test_size,dtype=int)
    for j in range(0,test_size):
        packet=received[j]
        message_rec=packet[n_state:]
        compare=(codes_message!=message_rec)
        hammdist=np.sum(compare,axis=1)
        index_min_dist=np.argmin(hammdist)
        message_predicted[j]=index_min_dist

    if(decoding_type=="HMM_no_delay"):
        startprob=np.full(S,1/S)
        T1=np.empty(S)
        prev_T1=np.empty(S)
        B=np.empty(S)

        packet=received[0]
        state_rec=packet[0:n_state]
        compare=(codes_state!=state_rec)
        hammdist=np.sum(compare,axis=1)
        B=np.power(p_predicted,hammdist)*np.power(1-p_predicted,n_state-hammdist)
        T1=startprob*B

        state_predicted[0]=np.argmax(T1)

        for j in range(1,test_size):
            prev_T1=np.array(T1)
            packet=received[j]
            state_rec=packet[0:n_state]
            compare=(codes_state!=state_rec)
            hammdist=np.sum(compare,axis=1)
            B=np.power(p_predicted,hammdist)*np.power(1-p_predicted,n_state-hammdist)
            for l in range(0,S):
                index_max_apriori=np.argmax(prev_T1*trans_mat_predicted[:,l])
                T1[l]=prev_T1[index_max_apriori]*trans_mat_predicted[index_max_apriori,l]*B[l]

            if(T1[0]<10**(-100)):
                T1=T1*10**100

            state_predicted[j]=np.argmax(T1)

        return state_predicted,message_predicted

    if(decoding_type=="HMM_delay"):
        startprob=np.full(S,1/S)
        T1=np.empty((S,test_size))
        Z=np.empty(test_size,dtype=int)
        T2=np.empty((S,test_size),dtype=int)
        B=np.empty(S)

        packet=received[0]
        state_rec=packet[0:n_state]
        compare=(codes_state!=state_rec)
        hammdist=np.sum(compare,axis=1)
        B=np.power(p_predicted,hammdist)*np.power(1-p_predicted,n_state-hammdist)


        T1[:,0]=startprob*B
        T2[:,0]=0

        for j in range(1,test_size):
            packet=received[j]
            state_rec=packet[0:n_state]
            compare=(codes_state!=state_rec)
            hammdist=np.sum(compare,axis=1)
            B=np.power(p_predicted,hammdist)*np.power(1-p_predicted,n_state-hammdist)

            for i in range(0,S):
                T2[i,j]=np.argmax(T1[:,j-1]*trans_mat_predicted[:,i])
                T1[i,j]=T1[T2[i,j],j-1]*trans_mat_predicted[T2[i,j],i]*B[i]

            if(T1[0,j]<10**(-100)):
                T1[:,j]=T1[:,j]*10**100

            Z[j]=np.argmax(T1[:,j])
            if(j>=delay):
                k=j
                while (k>(j-delay)):
                    Z[k-1]=T2[Z[k],k]
                    k-=1
                state_predicted[j-delay]=Z[j-delay]

        state_predicted[test_size-delay:test_size]=Z[test_size-delay:test_size]

        return state_predicted,message_predicted
    
#When the decoder has no initial knowlegde of the transition matrix and there is no separate training phase
#It learns the transition matrix as it decodes packets
def decoder_online_learning(S,M,n,received,p,codes,test_size,decoding_type,delay):
    if(decoding_type=="HMM_no_delay_MAP"):
        transitions_init=0.1
        state_predicted=np.empty(test_size,dtype=int)
        message_predicted=np.empty(test_size,dtype=int)

        transitions=np.full((S,S),transitions_init)
        trans_mat_predicted=transitions/np.sum(transitions,axis=1)
        startprob=np.full(S,1/S)

        B=np.empty(S)
        message_number=np.empty((S,test_size),dtype=int)
        T1=np.empty(S)
        prev_T1=np.empty(S)


        #First packet
        packet=received[0]
        compare=(codes!=packet)
        hammdist=np.sum(compare,axis=1)
        for l in range(0,S):
            index=np.argmin(hammdist[M*l:M*l+M])
            message_number[l,0]=index
            B[l]=np.sum(np.power(p,hammdist[M*l:M*l+M])*np.power((1-p),n-hammdist[M*l:M*l+M]))
        T1=startprob*B

        state_predicted[0]=np.argmax(T1)
        message_predicted[0]=message_number[state_predicted[0],0]


        for j in range(1,test_size):
            prev_T1=np.array(T1)
            packet=received[j]
            compare=(codes!=packet)
            hammdist=np.sum(compare,axis=1)
            for l in range(0,S):
                index=np.argmin(hammdist[M*l:M*l+M])
                message_number[l,j]=index
                B[l]=np.sum(np.power(p,hammdist[M*l:M*l+M])*np.power((1-p),n-hammdist[M*l:M*l+M]))
                apriori=np.sum(prev_T1*trans_mat_predicted[:,l])
                #apriori=np.sum(prev_T1*transitions[:,l])
                T1[l]=apriori*B[l]
            if(T1[0]<10**(-100)):
                T1=T1*10**100

            state_predicted[j]=np.argmax(T1)
            message_predicted[j]=message_number[state_predicted[j],j]    

            transitions[state_predicted[j-1],state_predicted[j]]+=1
            trans_mat_predicted=transitions/np.sum(transitions,axis=1)
        return state_predicted, message_predicted
    
    if(decoding_type=="HMM_delay_MAP"):
        transitions_init=0.1
        state_predicted=np.empty(test_size,dtype=int)
        message_predicted=np.empty(test_size,dtype=int)

        transitions=np.full((S,S),transitions_init)
        trans_mat_predicted=transitions/np.sum(transitions,axis=1)
        startprob=np.full(S,1/S)

        startprob=np.full(S,1/S)
        message_number=np.empty((S,test_size),dtype=int)
        forward=np.empty((S,test_size))
        backward=np.zeros((S,delay+1))
        B=np.empty((S,test_size))
        alpha=np.empty(S)

        packet=received[0]
        compare=(codes!=packet)
        hammdist=np.sum(compare,axis=1)
        for l in range(0,S):
            index=np.argmin(hammdist[M*l:M*l+M])
            message_number[l,0]=index
            B[l,0]=np.sum(np.power(p,hammdist[M*l:M*l+M])*np.power((1-p),n-hammdist[M*l:M*l+M]))

        forward[:,0]=startprob*B[:,0]


        for j in range(1,test_size):
            backward.fill(0)
            packet=received[j]
            compare=(codes!=packet)
            hammdist=np.sum(compare,axis=1)
            for l in range(0,S):
                index=np.argmin(hammdist[S*l:S*l+S])
                message_number[l,j]=index
                B[l,j]=np.sum(np.power(p,hammdist[S*l:S*l+S])*np.power((1-p),n-hammdist[S*l:S*l+S]))

            for i in range(0,S):
                apriori=np.sum(forward[:,j-1]*trans_mat_predicted[:,i])
                forward[i,j]=apriori*B[i,j]

            if(forward[0,j]<10**(-100)):
                forward[:,j]=forward[:,j]*10**100

            backward[:,delay].fill(1)
            for k in range(delay-1,-1,-1):
                for i in range(0,S):
                    alpha=np.squeeze(np.asarray(backward[:,k+1]))*np.squeeze(np.asarray(trans_mat_predicted[i,:]))
                    backward[i,k]=np.sum(alpha*B[:,j+k-delay+1])

            state_predicted[j-delay]=np.argmax(forward[:,j-delay]*backward[:,0])
            if(j-delay>=1):
                transitions[state_predicted[j-delay-1],state_predicted[j-delay]]+=1
            trans_mat_predicted=transitions/np.sum(transitions,axis=1)



        for i in range(test_size-delay,test_size):
            state_predicted[i]=np.argmax(forward[:,i]*backward[:,delay+i-test_size])

        for i in range(0,test_size):
            message_predicted[i]=message_number[state_predicted[i],i]
        return state_predicted,message_predicted
    

#Returns number of errors: first output is wrong state predictions, and second output is overall wrong predictions  
def errors(states,messages,state_predicted,message_predicted,test_size):
    error1=0
    for i in range(0,test_size):
        if(states[i]!=state_predicted[i] or messages[i]!=message_predicted[i]):
            error1+=1
    return np.sum(state_predicted!=states), error1
