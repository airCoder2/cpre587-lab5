#pragma once

#include "../Types.h"
#include "config.h"


#ifdef ZEDBOARD
#include "xparameters.h"
#include "xllfifo_hw.h" 
#endif

namespace ML
{
    // read one full packet (one full packet is one 4bytes)
    static i32 read_packet() {

        constexpr uint32_t RLR_PARTIAL = 0x80000000U;
        const uintptr_t base = XPAR_AXI_FIFO_0_BASEADDR;

        while (true) {
            while (Xil_In32(base + XLLF_RDFO_OFFSET) == 0);   // wait for data

            uint32_t rlr   = Xil_In32(base + XLLF_RLF_OFFSET);
            // basically MSB tells if packet if TLAST has been recieved (in our case from what I know we never get a packet
            // unless TLAST has been set, so we should never really get partial packets, meaning that every TVALID doesn't trigger
            // a transaction. But in our case TVALID is anyways tied to TLAST. So each accumulation is a single transaction)
            uint32_t bytes = rlr & 0x7FFFFFFFU;

            for (uint32_t i = 0; i < bytes; i += 4) {
                i32 recv_data = (i32)Xil_In32(base + XLLF_RDFD_OFFSET);
                return recv_data;
                //std::cout << "recv_data = " << recv_data << '\n';
            }
            // turning this off, since in our case there is no partial packets at all. if wait for data loop = 1, that means
            // packet is also complete
            // if (!(rlr & RLR_PARTIAL)) break;   // packet complete
        }
    }

    inline i32 concat_hex_to_i32(i8 a, i8 b) {
        // represent them as unsigned, shift a by eight to make it 15:8
        return (static_cast<uint8_t>(a) << 8) | static_cast<uint8_t>(b);
    }
}