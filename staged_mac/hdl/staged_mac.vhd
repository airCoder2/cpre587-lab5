-------------------------------------------------------------------------
-- Matthew Dwyer
-- Department of Electrical and Computer Engineering
-- Iowa State University
-------------------------------------------------------------------------


-- staged_mac.vhd
-------------------------------------------------------------------------
-- DESCRIPTION: This file contains a basic staged axi-stream mac unit. It
-- multiplies two integer values together and accumulates them.
--
-- NOTES:
-- 10/25/21 by MPD::Inital template creation
-- 9/5/25 by CWS::Minor changes to remove Qx.x
-------------------------------------------------------------------------

library work;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity staged_mac is
  generic(
      -- Parameters of mac
      C_DATA_WIDTH : integer := 8
    );
	port (
        ACLK	: in	std_logic;
		ARESETN	: in	std_logic;       

        -- AXIS slave data interface
		SD_AXIS_TREADY	: out	std_logic;
		SD_AXIS_TDATA	: in	std_logic_vector(31 downto 0);  -- Packed data input
		SD_AXIS_TLAST	: in	std_logic;
        SD_AXIS_TUSER   : in    std_logic;  -- Should we treat this first value in the stream as an inital accumulate value?
		SD_AXIS_TVALID	: in	std_logic;
        SD_AXIS_TID     : in    std_logic_vector(7 downto 0);

        -- AXIS master accumulate result out interface
		MO_AXIS_TVALID	: out	std_logic;
		MO_AXIS_TDATA	: out	std_logic_vector(31 downto 0);
		MO_AXIS_TLAST	: out	std_logic;
		MO_AXIS_TREADY	: in	std_logic;
		MO_AXIS_TID     : out   std_logic_vector(7 downto 0)
    );

-- ME: just letting the synthesis tool know that ACLK is clock, so it can route it diffrently
attribute SIGIS : string; 
attribute SIGIS of ACLK : signal is "Clk"; 

end staged_mac;


architecture behavioral of staged_mac is

    -- MY SIGNALS: 
    signal s_last_in_reg_out : std_logic;
    signal s_data_in_reg_out : std_logic_vector(31 downto 0);
    signal s_valid_in_reg_out: std_logic;

    signal s_accumulator_reg_out : std_logic_vector(31 downto 0);
    signal s_delayed_last_in: std_logic;

    signal s_adder_operand_b : std_logic_vector(31 downto 0);
    signal s_adder_out       : std_logic_vector(31 downto 0);
    signal s_mult_out        : std_logic_vector(C_DATA_WIDTH*2-1 downto 0);
    signal s_mult_out_extended : std_logic_vector(31 downto 0);

    signal s_ready_out : std_logic;
    signal s_expected_bias : std_logic;

    attribute use_dsp : string;
    attribute use_dsp of s_mult_out  : signal is "yes";
--    attribute use_dsp of s_adder_out : signal is "yes";


begin
    -- Assignments

    MO_AXIS_TDATA  <= s_accumulator_reg_out; -- accumulator's reg_out is the output data
    MO_AXIS_TVALID <= s_delayed_last_in;
    MO_AXIS_TLAST  <= s_delayed_last_in;  -- just never assign last keep the loop up

    -- Keep accumulating until accumulate data is valid (completed accumulation) and the slave is not ready to consume
    s_ready_out <= not s_delayed_last_in or MO_AXIS_TREADY;
    SD_AXIS_TREADY <= s_ready_out;

    -- if last was 1, then don't add the old value of accumulate again
    --s_adder_operand_a <= s_accumulator_reg_out when (s_delayed_last_in = '0') else (others => '0'); 
    
    -- multiply (15:8) and (7:0) of incoming data, if data is valid
    -- s_mult_out <= std_logic_vector(signed(s_data_in_reg_out(C_DATA_WIDTH*2-1 downto C_DATA_WIDTH)) * signed(s_data_in_reg_out(C_DATA_WIDTH-1 downto 0)));
    s_mult_out <= std_logic_vector(signed(s_data_in_reg_out((C_DATA_WIDTH + 8)-1 downto 8)) * signed(s_data_in_reg_out(C_DATA_WIDTH-1 downto 0)));
    
    -- THis is for sign extending the multiplicaiton
    with s_mult_out(C_DATA_WIDTH*2-1) select
        s_mult_out_extended <= (31 downto C_DATA_WIDTH*2 => '0') & s_mult_out when '0',
                               (31 downto C_DATA_WIDTH*2 => '1') & s_mult_out when '1',
                               (others => '0') when others;

    -- if data is not valid, then add 0 to current sum
    s_adder_operand_b <= s_mult_out_extended
                         when (s_valid_in_reg_out = '1') else (others => '0'); 
    
    -- This should work if we can guarantee that we always load the bias one clock cycle after tlast
     s_adder_out <= std_logic_vector(signed(s_accumulator_reg_out) + signed(s_adder_operand_b));
    -- s_adder_out <= std_logic_vector(unsigned(s_adder_operand_a) + unsigned(s_adder_operand_b));


    

    process(ACLK) is
    begin
        if rising_edge(ACLK) then
            if ARESETN = '0' then

                s_last_in_reg_out <= '0';
                s_delayed_last_in <= '0';
                s_valid_in_reg_out <= '0';
                s_expected_bias   <= '1'; -- the first valid transaction should be the bias, so set it to 1 on reset

                s_data_in_reg_out <= (others => '0');
                s_accumulator_reg_out <= (others => '0'); -- set the accumulator to 0

            elsif s_ready_out = '1' then
                s_last_in_reg_out <= SD_AXIS_TLAST;
                s_delayed_last_in <= s_last_in_reg_out and s_valid_in_reg_out;

                s_valid_in_reg_out <= SD_AXIS_TVALID;
                s_data_in_reg_out <= SD_AXIS_TDATA;

                -- if data is valid, then set last_state to expected bias.
                -- This way, we know next valid data is going to be bias
                if (s_valid_in_reg_out = '1') then
                    s_expected_bias <= s_last_in_reg_out;
                end if;
                
                -- if expected_bias was set to 1, and data is valid, then load bias.
                -- Otherwise store adder_out 
                if (s_expected_bias = '1' and s_valid_in_reg_out = '1') then
                    s_accumulator_reg_out <= s_data_in_reg_out;
                else
                    s_accumulator_reg_out <= s_adder_out;
                end if;

            end if;
        end if;
    end process;

end architecture behavioral;
