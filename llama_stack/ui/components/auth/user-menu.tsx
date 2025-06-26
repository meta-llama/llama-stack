"use client";

import React from "react";
import { useAuth } from "@/contexts/auth-context";
import { useAuthConfig } from "@/hooks/use-auth-config";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { User, LogOut, Key, Clock } from "lucide-react";
import { getAuthToken, isTokenExpired } from "@/lib/auth";
import { toast } from "sonner";

export function UserMenu() {
  const { user, logout, isAuthenticated } = useAuth();
  const { isAuthConfigured } = useAuthConfig();

  const handleCopyToken = async () => {
    const token = getAuthToken();
    if (!token) {
      toast.error("No authentication token found");
      return;
    }

    try {
      await navigator.clipboard.writeText(token);
      toast.success("API token copied to clipboard");
    } catch (error) {
      toast.error("Failed to copy token to clipboard");
    }
  };

  const getTokenExpiryInfo = () => {
    const token = getAuthToken();
    if (!token) return null;

    try {
      const payload = JSON.parse(atob(token.split(".")[1]));
      const exp = payload.exp;
      if (!exp) return null;

      const expiryDate = new Date(exp * 1000);
      const now = new Date();
      const hoursRemaining = Math.max(
        0,
        (expiryDate.getTime() - now.getTime()) / (1000 * 60 * 60),
      );

      if (hoursRemaining < 1) {
        return `Expires in ${Math.round(hoursRemaining * 60)} minutes`;
      } else if (hoursRemaining < 24) {
        return `Expires in ${Math.round(hoursRemaining)} hours`;
      } else {
        return `Expires in ${Math.round(hoursRemaining / 24)} days`;
      }
    } catch {
      return null;
    }
  };

  if (!isAuthenticated || !user) {
    if (!isAuthConfigured) {
      return (
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="inline-block">
              <Button variant="outline" size="sm" disabled>
                Sign In
              </Button>
            </span>
          </TooltipTrigger>
          <TooltipContent>
            <p>Authentication is not configured on this server</p>
          </TooltipContent>
        </Tooltip>
      );
    }

    return (
      <Button variant="outline" size="sm" asChild>
        <a href="/login">Sign In</a>
      </Button>
    );
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="icon" className="rounded-full">
          {user.avatar_url ? (
            <img
              src={user.avatar_url}
              alt={user.name || user.username}
              className="h-8 w-8 rounded-full"
            />
          ) : (
            <User className="h-4 w-4" />
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuLabel>
          <div className="flex flex-col space-y-1">
            <p className="text-sm font-medium leading-none">
              {user.name || user.username}
            </p>
            {user.email && (
              <p className="text-xs leading-none text-muted-foreground">
                {user.email}
              </p>
            )}
          </div>
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={handleCopyToken} className="cursor-pointer">
          <Key className="mr-2 h-4 w-4" />
          <span>Copy API Token</span>
        </DropdownMenuItem>
        {getTokenExpiryInfo() && (
          <DropdownMenuItem disabled className="text-xs text-muted-foreground">
            <Clock className="mr-2 h-3 w-3" />
            <span>{getTokenExpiryInfo()}</span>
          </DropdownMenuItem>
        )}
        <DropdownMenuSeparator />
        {user.organizations && user.organizations.length > 0 && (
          <>
            <DropdownMenuLabel className="text-xs text-muted-foreground">
              Organizations
            </DropdownMenuLabel>
            {user.organizations.map((org) => (
              <DropdownMenuItem key={org} className="text-sm">
                {org}
              </DropdownMenuItem>
            ))}
            <DropdownMenuSeparator />
          </>
        )}
        <DropdownMenuItem onClick={logout} className="cursor-pointer">
          <LogOut className="mr-2 h-4 w-4" />
          <span>Log out</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
