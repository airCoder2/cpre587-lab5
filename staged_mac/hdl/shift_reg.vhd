library work;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity shift_reg is
  generic (N_REGS : integer := 4);
	port (
        i_clk	 : in	std_logic;
        i_reset  : in std_logic;
		i_CE	 : in	std_logic;       
        i_data   : in std_logic;

        o_data   : out std_logic);
        
        -- ME: just letting the synthesis tool know that ACLK is clock, so it can route it diffrently
    attribute SIGIS : string; 
    attribute SIGIS of i_clk : signal is "Clk"; 
end shift_reg;



architecture behavioral of shift_reg is

    signal s_shift_reg : std_logic_vector(N_REGS-1 downto 0);

    -- assign the last bit to the output


begin
    
    o_data <= s_shift_reg(N_REGS-1);
    
    process(i_clk) is
    begin
        if rising_edge(i_clk) then
            if i_reset = '0' then
            
                s_shift_reg <= (others => '0');
        
            elsif i_CE = '1' then
            
                s_shift_reg(0) <= i_data;

                for i in 0 to N_REGS-2 loop
                    s_shift_reg(i+1) <= s_shift_reg(i);
                end loop;
                
            end if;

        end if;
    end process;

end architecture behavioral;

