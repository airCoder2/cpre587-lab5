library work;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity slave_interface is
	port (
        ACLK	: in	std_logic;
		ARESETN	: in	std_logic;       

        -- AXIS master accumulate result out interface
		MO_AXIS_TVALID	: out	std_logic;
		MO_AXIS_TDATA	: out	std_logic_vector(31 downto 0);
		MO_AXIS_TLAST	: out	std_logic;
		MO_AXIS_TREADY	: in	std_logic;
		MO_AXIS_TID     : out   std_logic_vector(7 downto 0);

        -- My interfaces
        slave_ready_out : out std_logic;
        dsp_acc_in : in std_logic_vector(31 downto 0);
        acc_valid_in : in std_logic
    );

-- ME: just letting the synthesis tool know that ACLK is clock, so it can route it diffrently
attribute SIGIS : string; 
attribute SIGIS of ACLK : signal is "Clk"; 

end slave_interface;


architecture behavioral of slave_interface is

    signal s_acc_reg_out : std_logic_vector(31 downto 0);
    signal s_acc_valid_reg_out : std_logic;
    signal s_slave_ready : std_logic;

begin

    MO_AXIS_TVALID <= s_acc_valid_reg_out;
    MO_AXIS_TDATA  <= s_acc_reg_out;
    MO_AXIS_TLAST  <= s_acc_valid_reg_out;

    s_slave_ready <= not s_acc_valid_reg_out or MO_AXIS_TREADY;
    slave_ready_out <= s_slave_ready;

    process(ACLK) is
    begin
        if rising_edge(ACLK) then
            if ARESETN = '0' then

                s_acc_reg_out <= (others => '0');
                s_acc_valid_reg_out <= '0';

            elsif s_slave_ready = '1' then
                s_acc_reg_out <= dsp_acc_in;
                s_acc_valid_reg_out <= acc_valid_in;
            end if;
        end if;
    end process;

end architecture behavioral;
