# BEGIN OF LICENSE NOTE
# This file is part of msgsr_net.
# Copyright (c) 2022, Steve Ahlswede, TU Berlin,
# ahlswede@tu-berlin.de
#
# TreeSat is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with msgsr_net. If not, see <https://www.gnu.org/licenses/>.
# END OF LICENSE NOTE

__all__ = [
    "__title__",
    "__summary__",
    "__uri__",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__copyright__",
]

__version__ = '0.1.0'

__title__ = "msgsr_net"
__summary__ = "A Python package for multi-label tree species classification using multi-modal data."
__uri__ = "https://gitlab.de"

__author__ = "Steve Ahlswede"
__email__ = "ahlswedes@gmail.com"

__license__ = "GPLv3+"
__copyright__ = "2022, %s" % __author__


def version():
    """Get the version of msgsr_net.
    Returns
    -------
    str
        Version specification.
    """
    return __version__